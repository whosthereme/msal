import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE)]
import tc_lodo
import numpy as np
import torch

PROTOCOL = "review_20261116_span_pred_v1"
ALPHAS = dict(lodo_xy6=0.0, lodo_xy6id05=0.5, lodo_xy6id1=1.0, lodo_xy6id2=2.0)
ARMS_RUN = ("lodo_xy6", "lodo_xy6id05", "lodo_xy6id1", "lodo_xy6id2")


def register():
  arms = tc_lodo.register()
  from msal_ext import run_nl
  from msal_ext.nl import FiLMNonlinearReadout
  from lib.pool_v5 import build_pool
  direction_of = dict((t.task_id, t.direction_16) for t in build_pool())
  xy6 = arms.DESCRIPTORS["lodo_xy6"]

  def make(alpha):
    def fn(ids):
      onehot = np.zeros((len(ids), 16))
      for i, t in enumerate(ids):
        onehot[i, direction_of[int(t)]] = 1.0
      return np.concatenate([xy6(ids), alpha * onehot], axis=1)
    return fn

  for name in ARMS_RUN[1:]:
    fn = make(ALPHAS[name])
    arms.DESCRIPTORS[name] = fn
    arms.ARMS[name] = dict(family="film", descriptor=name, reg=False)
    run_nl.BUILDERS[name] = (lambda f: lambda config, ids: FiLMNonlinearReadout(config, ids, f))(fn)
    run_nl.CONDITIONED[name] = ("film_gamma", "film_beta")
  return arms


def iota(D_fit, d):
  p = (np.linalg.pinv(D_fit) @ D_fit) @ d
  return float(p @ p / (d @ d))


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
  from msal_exp.data_lang import SPLIT_SELECTION, load_lang_v5
  from lib.frozen_data import DeviceCache
  from lib.pool_v5 import build_pool, source_task_ids
  from lib.structured_field import channel_weights
  from run_e35_1_zero_shot_code_loto import NOISE_DRAWS

  SEED, device = args.seed, args.device
  run_arms = [a for a in args.arms.split(",") if a]
  assert all(a in ALPHAS for a in run_arms), run_arms
  assert e2.TRAIN_STEPS == 12_800
  if args.smoke_steps:
    e2.TRAIN_STEPS = int(args.smoke_steps)
    args.output = args.output.with_name(args.output.name + "_SMOKE")
  args.output.mkdir(parents=True, exist_ok=True)
  path = args.output / "folds.jsonl"
  done = set()
  if path.exists():
    done = set((int(r["seed"]), r["arm"], int(r["held_direction"])) for r in map(json.loads, filter(str.strip, path.read_text().splitlines())))
  data = load_lang_v5("blind", verify_hash=False)
  cache = DeviceCache(data, device)
  data.state_features = None
  all_ids = [int(t) for t in source_task_ids()]
  direction_of = dict((t.task_id, t.direction_16) for t in build_pool())
  weights = channel_weights(int(data.action_chunks.shape[2]), dead_dims=e2.DEAD_ACTION_DIMS, device=device)
  info = dict(protocol=PROTOCOL, seed=SEED, arms=run_arms, smoke=bool(args.smoke_steps), lane=args.lane,
    environment=run_loto.environment(), repo_code_sha256=run_loto.code_hashes(), started_unix=time.time())
  (args.output / ("run_info_%d.json" % int(time.time()))).write_text(json.dumps(info, indent=1))

  def run_direction(arm, direction):
    held = [t for t in all_ids if direction_of[t] == direction]
    fit_ids = [t for t in all_ids if direction_of[t] != direction]
    started = time.time()
    model, _ = run_loto.train_arm(arm, seed=SEED, fit_ids=fit_ids, data=data, cache=cache, device=device)
    model.eval()
    train_seconds = time.time() - started
    D_fit = np.asarray(arms.DESCRIPTORS[arm](fit_ids), dtype=np.float64)
    records = []
    for task in held:
      flow = run_loto.score_forward(model, arm, held=task, seed=SEED, data=data, cache=cache, device=device)
      rows = data.rows_for_task(task, SPLIT_SELECTION)
      d_np = np.asarray(arms.DESCRIPTORS[arm]([task]), dtype=np.float64)
      d = torch.tensor(d_np, dtype=torch.float32, device=device)
      action = ideal.action_error(model, rows, d, cache, SEED, weights, NOISE_DRAWS, run_loto.DRAW_STRIDE)
      record = dict(arm=arm, seed=SEED, held_direction=direction, held_task=task, n_fit=len(fit_ids),
        train_steps=int(e2.TRAIN_STEPS), flow_selection=flow["main_selection"],
        flow_robustness=flow["main_robustness"], action_selection=action, rows=int(len(rows)),
        train_seconds=train_seconds)
      record.update(run_loto.parameter_report(model, arm))
      record.update(alpha=ALPHAS[arm], iota=iota(D_fit, d_np[0]), lane=args.lane, protocol=PROTOCOL)
      records.append(record)
    del model
    torch.cuda.empty_cache()
    return records, time.time() - started

  for direction in [int(x) for x in args.directions.split(",") if x.strip()]:
    for arm in run_arms:
      if (SEED, arm, direction) in done:
        continue
      if not args.smoke_steps and (BASE / "STOP_SPAN_PRED").exists():
        print("STOP_SPAN_PRED present", flush=True)
        return
      records, seconds = run_direction(arm, direction)
      with path.open("a") as handle:
        for r in records:
          handle.write(json.dumps(r) + "\n")
      print("[span %s] seed=%d direction=%2d tasks=%d %.0fs" % (arm, SEED, direction, len(records), seconds), flush=True)
  print("span-pred done", flush=True)


if __name__ == "__main__":
  main()
