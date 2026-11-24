import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE)]
import tc_combo
import tc_lodo
import numpy as np
import torch

ARMS_RUN = ("lodo_sat", "lodo_xy6")
PROTOCOL = "review_20261114_lodo_adapt_v1"
arms = tc_lodo.register()
from msal_ext import hyb, ideal
from msal_ext.hyb import K, SHOTS, FEWSHOT_OFFSET, e2, held_episode_rows, mean_mse, run_loto
from msal_exp.data_lang import LANG_NPZ, SPLIT_SELECTION, load_lang_v5
from lib.frozen_data import DeviceCache
from lib.pool_v5 import build_pool, source_task_ids
from lib.structured_field import channel_weights
from run_e35_1_zero_shot_code_loto import NOISE_DRAWS, ORACLE_DRAWS, weighted_ls_code


def adapt_task(R, lib, arm, held, seed, data, cache, episodes, device, weights):
  def scored(d):
    return dict((name, hyb.make_blocks(R, lib, cache, data.rows_for_task(held, split), d,
      seed=seed + getattr(run_loto, offset), draws=NOISE_DRAWS)) for name, split, offset in tc_combo.SPLITS)

  sel_rows = data.rows_for_task(held, SPLIT_SELECTION)

  def action(d, z):
    return ideal.action_error(tc_combo.Adapted(R, lib, "film_nl", d, z), sel_rows, d, cache, seed, weights,
      NOISE_DRAWS, run_loto.DRAW_STRIDE)

  d_star = torch.tensor(arms.DESCRIPTORS[arm]([held]), dtype=torch.float32, device=device)
  zero = torch.zeros(K, dtype=torch.float64, device=device)
  star = scored(d_star)
  out = dict(zero_shot=dict((s, mean_mse(b, zero, weights)) for s, b in star.items()), inversion={}, combination={},
    action=dict(zero_shot=action(d_star, None)))
  for k in SHOTS:
    rows = held_episode_rows(episodes, data, held, k)
    demo = hyb.make_blocks(R, lib, cache, rows, d_star, seed=seed + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
    d_inv = hyb.invert_descriptor(R, demo, d_star, weights)
    demo_inv = hyb.make_blocks(R, lib, cache, rows, d_inv, seed=seed + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
    z = weighted_ls_code(demo_inv, weights)
    inv = scored(d_inv)
    out["inversion"][str(k)] = dict((s, mean_mse(b, zero, weights)) for s, b in inv.items())
    out["combination"][str(k)] = dict((s, mean_mse(b, z, weights)) for s, b in inv.items())
    out["combination"][str(k)].update(code_norm=float(z.norm()), descriptor_shift=float((d_inv - d_star).norm()),
      demo_rows=int(len(rows)))
    if k in tc_combo.ACTION_K:
      out["action"]["inversion_%d" % k] = action(d_inv, None)
      out["action"]["combination_%d" % k] = action(d_inv, z)
  return out


def run_unit(arm, direction, seed, data, cache, episodes, device, weights, all_ids, direction_of):
  started = time.time()
  held = [t for t in all_ids if direction_of[t] == direction]
  fit_ids = [t for t in all_ids if direction_of[t] != direction]
  R, _ = run_loto.train_arm(arm, seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
  R.eval()
  flows = dict((t, run_loto.score_forward(R, arm, held=t, seed=seed, data=data, cache=cache, device=device)) for t in held)
  lib, stage2 = hyb.train_library(R, seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
  records = []
  for t in held:
    r = dict(arm=arm, seed=seed, held_direction=direction, held_task=t, n_fit=len(fit_ids),
      score_forward=dict(selection=flows[t]["main_selection"], robustness=flows[t]["main_robustness"]),
      stage2_final_loss=stage2)
    r.update(adapt_task(R, lib, arm, t, seed, data, cache, episodes, device, weights))
    records.append(r)
  seconds = time.time() - started
  for r in records:
    r["unit_seconds"] = seconds
  del R, lib
  torch.cuda.empty_cache()
  return records


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--directions", required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--smoke-steps", type=int, default=0)
  parser.add_argument("--lane", default="")
  args = parser.parse_args()
  assert e2.TRAIN_STEPS == 12_800
  if args.smoke_steps:
    e2.TRAIN_STEPS = int(args.smoke_steps)
    args.output = args.output.with_name(args.output.name + "_SMOKE")
  args.output.mkdir(parents=True, exist_ok=True)
  path = args.output / "folds.jsonl"
  done = set()
  if path.exists():
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    done = set((int(r["seed"]), r["arm"], int(r["held_direction"])) for r in rows)
  device = args.device
  data = load_lang_v5("blind", verify_hash=False)
  episodes = np.load(LANG_NPZ, mmap_mode="r")["episode_indices"].astype(np.int64)
  cache = DeviceCache(data, device)
  data.state_features = None
  weights = channel_weights(int(data.action_chunks.shape[2]), dead_dims=e2.DEAD_ACTION_DIMS, device=device)
  all_ids = [int(t) for t in source_task_ids()]
  direction_of = dict((t.task_id, t.direction_16) for t in build_pool())
  info = dict(protocol=PROTOCOL, seed=args.seed, smoke=bool(args.smoke_steps), lane=args.lane,
    environment=run_loto.environment(), repo_code_sha256=run_loto.code_hashes(), started_unix=time.time())
  (args.output / ("run_info_%d.json" % int(time.time()))).write_text(json.dumps(info, indent=1))
  for direction in [int(d) for d in args.directions.split(",") if d.strip()]:
    for arm in ARMS_RUN:
      if (args.seed, arm, direction) in done:
        continue
      if not args.smoke_steps and (BASE / "STOP_LODO_ADAPT").exists():
        print("STOP_LODO_ADAPT present", flush=True)
        return
      try:
        records = run_unit(arm, direction, args.seed, data, cache, episodes, device, weights, all_ids, direction_of)
      except FloatingPointError:
        records = run_unit(arm, direction, args.seed, data, cache, episodes, device, weights, all_ids, direction_of)
        for r in records:
          r["retrained_after_nan"] = True
      with path.open("a") as handle:
        for r in records:
          r["lane"] = args.lane
          handle.write(json.dumps(r) + "\n")
      print("[lodo-adapt %s] seed=%d direction=%2d tasks=%d %.0fs" % (arm, args.seed, direction, len(records),
        records[0]["unit_seconds"]), flush=True)
  print("lodo-adapt done", flush=True)


if __name__ == "__main__":
  main()
