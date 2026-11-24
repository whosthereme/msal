import argparse
import json
import socket
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE)]
import tc_combo
import numpy as np
import torch

from msal_ext import ideal, l40, run_nl
from msal_ext.hyb import e2, run_loto
from msal_ext.nl import FiLMNonlinearReadout
from lib import g39_data
from lib.frozen_data import DeviceCache
from lib.structured_field import channel_weights
from run_e35_1_zero_shot_code_loto import NOISE_DRAWS

PROTOCOL = "review_20261119_text_v1"
ARM = "l40_text_nl"
RELATIVE_TOLERANCE = 1e-9
FOLD = {}
assert e2.DEAD_ACTION_DIMS == ()


def text_descriptor(task_ids):
  return np.stack([FOLD["D"][FOLD["index"][int(t)]] for t in task_ids]).astype(np.float64)


run_nl.arms.DESCRIPTORS["l40_text"] = text_descriptor
run_nl.arms.ARMS[ARM] = dict(family="film", descriptor="l40_text", reg=False)
run_nl.BUILDERS[ARM] = lambda config, ids: FiLMNonlinearReadout(config, ids, text_descriptor)
run_nl.CONDITIONED[ARM] = ("film_gamma", "film_beta")


def projector(D_fit):
  _, s, vt = np.linalg.svd(np.asarray(D_fit, dtype=np.float64), full_matrices=False)
  rank = int((s > RELATIVE_TOLERANCE * s[0]).sum())
  basis = vt[:rank]
  return basis.T @ basis, rank


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--held-tasks", required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--text-dir", type=Path, default=BASE / "trunkcond" / "text")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--smoke-steps", type=int, default=0)
  parser.add_argument("--lane", default="")
  args = parser.parse_args()
  sys.path.insert(0, str(args.text_dir))
  import text_desc
  ids, U = text_desc.load(args.text_dir / "text_embeddings.npz", "last")
  assert sorted(ids) == sorted(l40.INSTRUCTIONS), "embedding file does not cover the 40 tasks"
  assert e2.TRAIN_STEPS == 12_800
  if args.smoke_steps:
    e2.TRAIN_STEPS = int(args.smoke_steps)
    args.output = args.output.with_name(args.output.name + "_SMOKE")
  args.output.mkdir(parents=True, exist_ok=True)
  path = args.output / "folds.jsonl"
  done = set()
  if path.exists():
    lines = [x for x in path.read_text().splitlines() if x.strip()]
    done = set((int(r["seed"]), int(r["held_task"])) for r in map(json.loads, lines))
  seed, device, host = args.seed, args.device, socket.gethostname()
  data = g39_data.load_g39_data(verify_hash=False)
  cache = DeviceCache(data, device)
  data.state_features = np.empty((0, g39_data.FROZEN_FEATURE_DIM), dtype=np.float16)
  weights = channel_weights(int(data.action_chunks.shape[2]), dead_dims=e2.DEAD_ACTION_DIMS, device=device)
  info = dict(protocol=PROTOCOL, seed=seed, smoke=bool(args.smoke_steps), lane=args.lane, host=host,
    environment=run_loto.environment(), repo_code_sha256=run_loto.code_hashes(),
    runner_sha256=run_loto.file_sha(Path(__file__).resolve()),
    embeddings_sha256=run_loto.file_sha(args.text_dir / "text_embeddings.npz"), started_unix=time.time())
  (args.output / ("run_info_%d.json" % int(time.time()))).write_text(json.dumps(info, indent=1))

  def flow_with(model, held, d):
    out = {}
    for name, split, offset in (("selection", g39_data.SPLIT_VAL, run_loto.SELECTION_OFFSET),
        ("robustness", g39_data.SPLIT_TEST, run_loto.ROBUSTNESS_OFFSET)):
      rows = data.rows_for_task(held, split)
      out[name] = run_loto._forward_mse(model, ARM, rows=rows, seed=seed, offset=offset, cache=cache, device=device,
        descriptor=d)
    return out

  def run_fold(held):
    started = time.time()
    FOLD.update(D=text_desc.fold_matrix(ids, U, held), index=dict((t, i) for i, t in enumerate(ids)))
    fit_ids = [t for t in sorted(l40.INSTRUCTIONS) if t != held]
    model, _ = run_loto.train_arm(ARM, seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
    model.eval()
    train_seconds = time.time() - started
    stored = run_loto.score_forward(model, ARM, held=held, seed=seed, data=data, cache=cache, device=device)
    d_np = text_descriptor([held])
    P, rank = projector(text_descriptor(fit_ids))
    p_np = (P @ d_np[0])[None, :]
    m_np = np.zeros_like(d_np)
    m_np[0, 0] = 1.0
    d, p, m = (torch.tensor(v, dtype=torch.float32, device=device) for v in (d_np, p_np, m_np))
    rows = data.rows_for_task(held, g39_data.SPLIT_VAL)
    raw, proj, mean = flow_with(model, held, d), flow_with(model, held, p), flow_with(model, held, m)

    def act(v):
      return ideal.action_error(model, rows, v, cache, seed, weights, NOISE_DRAWS, run_loto.DRAW_STRIDE)

    record = dict(arm="text_nl", seed=seed, held_task=held, train_steps=int(e2.TRAIN_STEPS),
      flow_selection=stored["main_selection"], flow_robustness=stored["main_robustness"], action_selection=act(d),
      raw_matches_stored_scorer=bool(raw["selection"] == stored["main_selection"]),
      proj_flow_selection=proj["selection"], proj_flow_robustness=proj["robustness"], proj_action_selection=act(p),
      mean_flow_selection=mean["selection"], mean_flow_robustness=mean["robustness"], mean_action_selection=act(m),
      iota=float(p_np[0] @ p_np[0] / (d_np[0] @ d_np[0])), span_rank=rank, descriptor_dim=int(d_np.shape[1]),
      descriptor_norm=float(np.linalg.norm(d_np[0])), projected_norm=float(np.linalg.norm(p_np[0])),
      rows=int(len(rows)), train_seconds=train_seconds, lane=args.lane, host=host, protocol=PROTOCOL)
    record.update(run_loto.parameter_report(model, ARM))
    record["seconds"] = time.time() - started
    del model
    torch.cuda.empty_cache()
    return record

  for held in [int(t) for t in args.held_tasks.split(",") if t.strip()]:
    if (seed, held) in done:
      continue
    if not args.smoke_steps and (BASE / "STOP_TEXT").exists():
      print("STOP_TEXT present", flush=True)
      return
    r = run_fold(held)
    with path.open("a") as handle:
      handle.write(json.dumps(r) + "\n")
    print("[text] seed=%d held=%d iota=%.3f flow=%.5f in-span=%.5f mean=%.5f act=%.5f %.0fs" % (seed, held, r["iota"],
      r["flow_selection"], r["proj_flow_selection"], r["mean_flow_selection"], r["action_selection"], r["seconds"]),
      flush=True)
  print("text unit done", flush=True)


if __name__ == "__main__":
  main()
