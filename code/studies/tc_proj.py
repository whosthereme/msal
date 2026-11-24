import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE)]
import tc_span_pred
import numpy as np
import torch

PROTOCOL = "review_20261118_proj_v1"
ARMS_RUN = ("lodo_sat", "lodo_xy6id2")
RELATIVE_TOLERANCE = 1e-9


def projector(D_fit):
  _, s, vt = np.linalg.svd(np.asarray(D_fit, dtype=np.float64), full_matrices=False)
  rank = int((s > RELATIVE_TOLERANCE * s[0]).sum())
  basis = vt[:rank]
  return basis.T @ basis, rank


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--directions", required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--arms", default=",".join(ARMS_RUN))
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--smoke-steps", type=int, default=0)
  parser.add_argument("--lane", default="")
  args = parser.parse_args()
  arms = tc_span_pred.register()
  from msal_ext import ideal
  from msal_ext.hyb import e2, run_loto
  from msal_exp.data_lang import SPLIT_ROBUSTNESS, SPLIT_SELECTION, load_lang_v5
  from lib.frozen_data import DeviceCache
  from lib.pool_v5 import build_pool, source_task_ids
  from lib.structured_field import channel_weights
  from run_e35_1_zero_shot_code_loto import NOISE_DRAWS

  SEED, device = args.seed, args.device
  run_arms = [a for a in args.arms.split(",") if a]
  assert all(a in ARMS_RUN for a in run_arms), run_arms
  assert e2.TRAIN_STEPS == 12_800
  if args.smoke_steps:
    e2.TRAIN_STEPS = int(args.smoke_steps)
    args.output = args.output.with_name(args.output.name + "_SMOKE")
  args.output.mkdir(parents=True, exist_ok=True)
  path = args.output / "folds.jsonl"
  done = set()
  if path.exists():
    lines = [x for x in path.read_text().splitlines() if x.strip()]
    done = set((int(r["seed"]), r["arm"], int(r["held_direction"])) for r in map(json.loads, lines))
  data = load_lang_v5("blind", verify_hash=False)
  cache = DeviceCache(data, device)
  data.state_features = None
  all_ids = [int(t) for t in source_task_ids()]
  direction_of = dict((t.task_id, t.direction_16) for t in build_pool())
  weights = channel_weights(int(data.action_chunks.shape[2]), dead_dims=e2.DEAD_ACTION_DIMS, device=device)
  info = dict(protocol=PROTOCOL, seed=SEED, arms=run_arms, smoke=bool(args.smoke_steps), lane=args.lane,
    environment=run_loto.environment(), repo_code_sha256=run_loto.code_hashes(), started_unix=time.time(),
    runner_sha256=run_loto.file_sha(Path(__file__).resolve()))
  (args.output / ("run_info_%d.json" % int(time.time()))).write_text(json.dumps(info, indent=1))

  def flow_with(model, arm, task, d):
    out = {}
    for name, split, offset in (("selection", SPLIT_SELECTION, run_loto.SELECTION_OFFSET),
        ("robustness", SPLIT_ROBUSTNESS, run_loto.ROBUSTNESS_OFFSET)):
      rows = data.rows_for_task(task, split)
      out[name] = run_loto._forward_mse(model, arm, rows=rows, seed=SEED, offset=offset, cache=cache,
        device=device, descriptor=d)
    return out

  def score_task(model, arm, task, P, rank, n_fit, direction, train_seconds):
    stored = run_loto.score_forward(model, arm, held=task, seed=SEED, data=data, cache=cache, device=device)
    rows = data.rows_for_task(task, SPLIT_SELECTION)
    d_np = np.asarray(arms.DESCRIPTORS[arm]([task]), dtype=np.float64)
    p_np = (P @ d_np[0])[None, :]
    d = torch.tensor(d_np, dtype=torch.float32, device=device)
    p = torch.tensor(p_np, dtype=torch.float32, device=device)
    raw = flow_with(model, arm, task, d)
    proj = flow_with(model, arm, task, p)
    act_raw = ideal.action_error(model, rows, d, cache, SEED, weights, NOISE_DRAWS, run_loto.DRAW_STRIDE)
    act_proj = ideal.action_error(model, rows, p, cache, SEED, weights, NOISE_DRAWS, run_loto.DRAW_STRIDE)
    record = dict(arm=arm, seed=SEED, held_direction=direction, held_task=task, n_fit=n_fit,
      train_steps=int(e2.TRAIN_STEPS), flow_selection=stored["main_selection"],
      flow_robustness=stored["main_robustness"], action_selection=act_raw,
      raw_matches_stored_scorer=bool(raw["selection"] == stored["main_selection"]),
      proj_flow_selection=proj["selection"], proj_flow_robustness=proj["robustness"],
      proj_action_selection=act_proj, rows=int(len(rows)), train_seconds=train_seconds,
      iota=float(p_np[0] @ p_np[0] / (d_np[0] @ d_np[0])), span_rank=rank, descriptor_dim=int(d_np.shape[1]),
      descriptor_norm=float(np.linalg.norm(d_np[0])), projected_norm=float(np.linalg.norm(p_np[0])),
      lane=args.lane, protocol=PROTOCOL)
    record.update(run_loto.parameter_report(model, arm))
    return record

  def run_direction(arm, direction):
    held = [t for t in all_ids if direction_of[t] == direction]
    fit_ids = [t for t in all_ids if direction_of[t] != direction]
    started = time.time()
    model, _ = run_loto.train_arm(arm, seed=SEED, fit_ids=fit_ids, data=data, cache=cache, device=device)
    model.eval()
    train_seconds = time.time() - started
    P, rank = projector(arms.DESCRIPTORS[arm](fit_ids))
    records = [score_task(model, arm, t, P, rank, len(fit_ids), direction, train_seconds) for t in held]
    del model
    torch.cuda.empty_cache()
    return records, time.time() - started

  for direction in [int(x) for x in args.directions.split(",") if x.strip()]:
    for arm in run_arms:
      if (SEED, arm, direction) in done:
        continue
      if not args.smoke_steps and (BASE / "STOP_PROJ").exists():
        print("STOP_PROJ present", flush=True)
        return
      records, seconds = run_direction(arm, direction)
      with path.open("a") as handle:
        for r in records:
          handle.write(json.dumps(r) + "\n")
      print("[proj %s] seed=%d direction=%2d tasks=%d iota=%.3f %.0fs" % (arm, SEED, direction, len(records),
        float(np.mean([r["iota"] for r in records])), seconds), flush=True)
  print("proj done", flush=True)


if __name__ == "__main__":
  main()
