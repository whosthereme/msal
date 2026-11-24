import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE)]
import tc_l40
import numpy as np
import torch

from tc_l40 import (K, SHOTS, FEWSHOT_OFFSET, ACTION_K, KINDS, NOISE_DRAWS, ORACLE_DRAWS, e2, g39_data, held_episode_rows,
  hyb, ideal, l40, mean_mse, run_loto, tc_combo, weighted_ls_code, words, DeviceCache, channel_weights)

PROTOCOL = "review_20261114_l40_zero_v1"
UNSEEN_TASKS = (1000, 1001, 1005, 1006, 1008, 1009, 1010, 1011, 1012, 1017, 1021, 1025, 1026, 1027, 1028, 1030, 1031)


def evaluate(R, lib, kind, held, seed, data, cache, device, weights, d0):
  head = kind == "film_nl"
  blocks = hyb.make_blocks if head else tc_combo.make_blocks_t
  invert = hyb.invert_descriptor if head else tc_combo.invert_t
  val_rows = data.rows_for_task(held, g39_data.SPLIT_VAL)
  zero = torch.zeros(K, dtype=torch.float64, device=device)

  def scored(d):
    return dict((name, blocks(R, lib, cache, data.rows_for_task(held, split), d,
      seed=seed + getattr(run_loto, offset), draws=NOISE_DRAWS)) for name, split, offset in l40.SPLITS)

  def action(d, z):
    return ideal.action_error(tc_combo.Adapted(R, lib, kind, d, z), val_rows, d, cache, seed, weights, NOISE_DRAWS,
      run_loto.DRAW_STRIDE)

  star = scored(d0)
  out = dict(zero_shot=dict((s, mean_mse(b, zero, weights)) for s, b in star.items()), inversion={}, combination={},
    action=dict(zero_shot=action(d0, None)))
  for k in SHOTS:
    rows = held_episode_rows(data.episode_indices, data, held, k)
    demo = blocks(R, lib, cache, rows, d0, seed=seed + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
    d_inv = invert(R, demo, d0, weights)
    demo_inv = blocks(R, lib, cache, rows, d_inv, seed=seed + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
    z = weighted_ls_code(demo_inv, weights)
    inv = scored(d_inv)
    out["inversion"][str(k)] = dict((s, mean_mse(b, zero, weights)) for s, b in inv.items())
    out["combination"][str(k)] = dict((s, mean_mse(b, z, weights)) for s, b in inv.items())
    out["combination"][str(k)].update(code_norm=float(z.norm()), descriptor_shift=float((d_inv - d0).norm()),
      demo_rows=int(len(rows)))
    if k in ACTION_K:
      out["action"]["inversion_%d" % k] = action(d_inv, None)
      out["action"]["combination_%d" % k] = action(d_inv, z)
  return out


def run_pair_head(kind, held, seed, data, cache, device, weights):
  started = time.time()
  arm = "l40_" + kind
  fit_ids = [t for t in sorted(l40.INSTRUCTIONS) if t != held]
  R, _ = run_loto.train_arm(arm, seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
  R.eval()
  zero_flow = run_loto.score_forward(R, arm, held=held, seed=seed, data=data, cache=cache, device=device)
  train = hyb.train_library if kind == "film_nl" else tc_combo.train_library_t
  lib, stage2 = train(R, seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
  d_star = torch.tensor(l40.word_descriptor([held]), dtype=torch.float32, device=device)
  seen_cols = (np.abs(l40.word_descriptor(fit_ids)).sum(axis=0) > 0).astype(np.float32)
  mask = torch.tensor(seen_cols, device=device)
  d_proj = d_star * mask
  seen = set().union(*[words(l40.INSTRUCTIONS[t]) for t in fit_ids])
  record = dict(arm=kind, held_task=held, seed=seed, unseen_words=sorted(words(l40.INSTRUCTIONS[held]) - seen),
    zeroed_entries=int(((d_star != 0) & (mask == 0)).sum()),
    score_forward=dict(selection=zero_flow["main_selection"], robustness=zero_flow["main_robustness"]),
    stage2_final_loss=stage2)
  record["raw"] = evaluate(R, lib, kind, held, seed, data, cache, device, weights, d_star)
  record["projected"] = evaluate(R, lib, kind, held, seed, data, cache, device, weights, d_proj)
  record["seconds"] = time.time() - started
  del R, lib
  torch.cuda.empty_cache()
  return record


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--held-tasks", required=True)
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
    done = set((int(r["seed"]), r["arm"], int(r["held_task"])) for r in rows)
  device = args.device
  data = g39_data.load_g39_data(verify_hash=False)
  cache = DeviceCache(data, device)
  data.state_features = np.empty((0, g39_data.FROZEN_FEATURE_DIM), dtype=np.float16)
  weights = channel_weights(int(data.action_chunks.shape[2]), dead_dims=e2.DEAD_ACTION_DIMS, device=device)
  info = dict(protocol=PROTOCOL, seed=args.seed, smoke=bool(args.smoke_steps), lane=args.lane,
    environment=run_loto.environment(), repo_code_sha256=run_loto.code_hashes(), started_unix=time.time())
  (args.output / ("run_info_%d.json" % int(time.time()))).write_text(json.dumps(info, indent=1))
  for held in [int(t) for t in args.held_tasks.split(",") if t.strip()]:
    assert held in UNSEEN_TASKS, held
    for kind in KINDS:
      if (args.seed, kind, held) in done:
        continue
      if not args.smoke_steps and (BASE / "STOP_L40Z").exists():
        print("STOP_L40Z present", flush=True)
        return
      try:
        record = run_pair_head(kind, held, args.seed, data, cache, device, weights)
      except FloatingPointError:
        record = run_pair_head(kind, held, args.seed, data, cache, device, weights)
        record["retrained_after_nan"] = True
      record["lane"] = args.lane
      with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
      print("[l40z %s] seed=%d held=%d %.0fs" % (kind, args.seed, held, record["seconds"]), flush=True)
  print("l40z done", flush=True)


if __name__ == "__main__":
  main()
