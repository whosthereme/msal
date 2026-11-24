import argparse
import json
import socket
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE)]
import tc_proj
import numpy as np
import torch

tc_span_pred = tc_proj.tc_span_pred
PROTOCOL = "trunk2_20261121_lodo_v1"
ARMS_RUN = ("lodo_plain", "lodo_tsat", "lodo_txy6", "lodo_txy6id2", "lodo_txy6id1", "lodo_txy6id05")
SOURCE = dict(lodo_tsat="lodo_sat", lodo_txy6id05="lodo_xy6id05", lodo_txy6id1="lodo_xy6id1", lodo_txy6id2="lodo_xy6id2")
ALPHA = dict(lodo_txy6=0.0, lodo_txy6id05=0.5, lodo_txy6id1=1.0, lodo_txy6id2=2.0)
PROJECTED = ("lodo_tsat", "lodo_txy6id2")


def register():
  arms = tc_span_pred.register()
  import tc_arms
  from msal_ext import run_nl
  from msal_ext.nl import PlainNonlinearReadout

  class TrunkFiLMFn(tc_arms.TrunkFiLMField):

    def __init__(self, config, source_task_ids, fn):
      PlainNonlinearReadout.__init__(self, config, source_task_ids)
      matrix = fn(source_task_ids)
      self.register_buffer("task_features", torch.tensor(matrix, dtype=torch.float32), persistent=True)
      self.trunk_gamma = tc_arms.bias_free(int(matrix.shape[1]), config.hidden_dim)
      self.trunk_beta = tc_arms.bias_free(int(matrix.shape[1]), config.hidden_dim)

  for name, source in SOURCE.items():
    fn = arms.DESCRIPTORS[source]
    arms.DESCRIPTORS[name] = fn
    arms.ARMS[name] = dict(family="film", descriptor=name, reg=False)
    run_nl.BUILDERS[name] = (lambda f: lambda config, ids: TrunkFiLMFn(config, ids, f))(fn)
    run_nl.CONDITIONED[name] = ("trunk_gamma", "trunk_beta")
  return arms


class NoDescriptor:
  def expand(self, n, _):
    return None


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
  arms = register()
  from msal_ext import ideal
  from msal_ext.hyb import e2, run_loto
  from msal_exp.data_lang import SPLIT_ROBUSTNESS, SPLIT_SELECTION, load_lang_v5
  from lib.frozen_data import DeviceCache
  from lib.pool_v5 import build_pool, source_task_ids
  from lib.structured_field import channel_weights
  from run_e35_1_zero_shot_code_loto import NOISE_DRAWS

  SEED, device, host = args.seed, args.device, socket.gethostname()
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
  info = dict(protocol=PROTOCOL, seed=SEED, arms=run_arms, smoke=bool(args.smoke_steps), lane=args.lane, host=host,
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
    record = dict(arm=arm, seed=SEED, held_direction=direction, held_task=task, n_fit=n_fit,
      train_steps=int(e2.TRAIN_STEPS), flow_selection=stored["main_selection"],
      flow_robustness=stored["main_robustness"], rows=int(len(rows)), train_seconds=train_seconds, lane=args.lane,
      host=host, protocol=PROTOCOL)
    if arm == "lodo_plain":
      record["action_selection"] = ideal.action_error(model, rows, NoDescriptor(), cache, SEED, weights, NOISE_DRAWS,
        run_loto.DRAW_STRIDE)
    else:
      d_np = np.asarray(arms.DESCRIPTORS[arm]([task]), dtype=np.float64)
      p_np = (P @ d_np[0])[None, :]
      d = torch.tensor(d_np, dtype=torch.float32, device=device)
      record["action_selection"] = ideal.action_error(model, rows, d, cache, SEED, weights, NOISE_DRAWS,
        run_loto.DRAW_STRIDE)
      record.update(iota=float(p_np[0] @ p_np[0] / (d_np[0] @ d_np[0])), span_rank=rank,
        descriptor_dim=int(d_np.shape[1]), alpha=ALPHA.get(arm))
      if arm in PROJECTED:
        p = torch.tensor(p_np, dtype=torch.float32, device=device)
        raw = flow_with(model, arm, task, d)
        proj = flow_with(model, arm, task, p)
        record.update(raw_matches_stored_scorer=bool(raw["selection"] == stored["main_selection"]),
          proj_flow_selection=proj["selection"], proj_flow_robustness=proj["robustness"],
          proj_action_selection=ideal.action_error(model, rows, p, cache, SEED, weights, NOISE_DRAWS,
            run_loto.DRAW_STRIDE))
    record.update(run_loto.parameter_report(model, arm))
    return record

  def run_direction(arm, direction):
    held = [t for t in all_ids if direction_of[t] == direction]
    fit_ids = [t for t in all_ids if direction_of[t] != direction]
    started = time.time()
    model, _ = run_loto.train_arm(arm, seed=SEED, fit_ids=fit_ids, data=data, cache=cache, device=device)
    model.eval()
    train_seconds = time.time() - started
    P, rank = (None, None) if arm == "lodo_plain" else tc_proj.projector(arms.DESCRIPTORS[arm](fit_ids))
    records = [score_task(model, arm, t, P, rank, len(fit_ids), direction, train_seconds) for t in held]
    del model
    torch.cuda.empty_cache()
    return records, time.time() - started

  for direction in [int(x) for x in args.directions.split(",") if x.strip()]:
    for arm in run_arms:
      if (SEED, arm, direction) in done:
        continue
      if not args.smoke_steps and (BASE / "STOP_TRUNK2").exists():
        print("STOP_TRUNK2 present", flush=True)
        return
      records, seconds = run_direction(arm, direction)
      with path.open("a") as handle:
        for r in records:
          handle.write(json.dumps(r) + "\n")
      print("[lodo_t %s] seed=%d direction=%2d tasks=%d %.0fs" % (arm, SEED, direction, len(records), seconds), flush=True)
  print("lodo_t unit done", flush=True)


if __name__ == "__main__":
  main()
