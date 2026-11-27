import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE)]
import tc_combo as C
import numpy as np
import torch

PROTOCOL = "review_20261127_penalty_v1"
SEED = 148021
ARM = "tfilm_nl"
SETTINGS = (("S0", 0.02, 0.001), ("S1", 0.0, 0.001), ("S2", 0.08, 0.001), ("S3", 0.02, 0.0), ("S4", 0.02, 0.004))


def run_task(held, data, cache, episodes, device, weights, done, path, lane):
  e2 = C.e2
  started = time.time()
  fit_ids = [t for t in C.source_task_ids() if t != held]
  R, _ = C.run_loto.train_arm(ARM, seed=SEED, fit_ids=fit_ids, data=data, cache=cache, device=device)
  R.eval()
  stage1_seconds = time.time() - started
  zero_flow = C.run_loto.score_forward(R, ARM, held=held, seed=SEED, data=data, cache=cache, device=device)
  d_star = torch.tensor(C.run_nl.arms.DESCRIPTORS["saturated"]([held]), dtype=torch.float32, device=device)
  zero = torch.zeros(C.K, dtype=torch.float64, device=device)
  sel_rows = data.rows_for_task(held, C.SPLIT_SELECTION)
  shot_rows = dict((k, C.held_episode_rows(episodes, data, held, k)) for k in C.SHOTS)
  inverted = {}
  for name, w_orth, w_group in SETTINGS:
    if (held, name) in done:
      continue
    t0 = time.time()
    e2.ORTHOGONALITY_WEIGHT, e2.GROUP_SPARSITY_WEIGHT = w_orth, w_group
    try:
      try:
        lib, stage2 = C.train_library_t(R, seed=SEED, fit_ids=fit_ids, data=data, cache=cache, device=device)
      except FloatingPointError:
        lib, stage2 = C.train_library_t(R, seed=SEED, fit_ids=fit_ids, data=data, cache=cache, device=device)
    except FloatingPointError:
      record = dict(protocol=PROTOCOL, arm=ARM, held_task=held, seed=SEED, setting=name, failed=True, lane=lane)
      with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
      continue
    finally:
      e2.ORTHOGONALITY_WEIGHT, e2.GROUP_SPARSITY_WEIGHT = 0.02, 0.001

    def scored(d):
      return dict((s, C.make_blocks_t(R, lib, cache, data.rows_for_task(held, split), d,
        seed=SEED + getattr(C.run_loto, offset), draws=C.NOISE_DRAWS)) for s, split, offset in C.SPLITS)

    def action(d, z):
      return C.ideal.action_error(C.Adapted(R, lib, ARM, d, z), sel_rows, d, cache, SEED, weights, C.NOISE_DRAWS,
        C.run_loto.DRAW_STRIDE)

    star = scored(d_star)
    slot_norms = lib.codes.detach().double().norm(dim=0).cpu().tolist()
    record = dict(protocol=PROTOCOL, arm=ARM, held_task=held, seed=SEED, setting=name, orth_weight=w_orth,
      group_weight=w_group, score_forward_selection=zero_flow["main_selection"], stage2_final_loss=stage2,
      slot_norms=slot_norms, zero_shot=dict((s, C.mean_mse(b, zero, weights)) for s, b in star.items()),
      inversion={}, combination={}, action={})
    for k in C.SHOTS:
      demo = C.make_blocks_t(R, lib, cache, shot_rows[k], d_star, seed=SEED + C.FEWSHOT_OFFSET, draws=C.ORACLE_DRAWS)
      if k not in inverted:
        inverted[k] = C.invert_t(R, demo, d_star, weights)
      d_inv = inverted[k]
      demo_inv = C.make_blocks_t(R, lib, cache, shot_rows[k], d_inv, seed=SEED + C.FEWSHOT_OFFSET,
        draws=C.ORACLE_DRAWS)
      z = C.weighted_ls_code(demo_inv, weights)
      inv = scored(d_inv)
      record["inversion"][str(k)] = dict((s, C.mean_mse(b, zero, weights)) for s, b in inv.items())
      record["combination"][str(k)] = dict((s, C.mean_mse(b, z, weights)) for s, b in inv.items())
      record["combination"][str(k)].update(code_norm=float(z.norm()), descriptor_shift=float((d_inv - d_star).norm()))
      if k in C.ACTION_K:
        record["action"]["combination_%d" % k] = action(d_inv, z)
    record["seconds"] = time.time() - t0
    record["stage1_seconds"] = stage1_seconds
    record["lane"] = lane
    with path.open("a") as handle:
      handle.write(json.dumps(record) + "\n")
    print("[penalty %s] held=%2d %.0fs" % (name, held, record["seconds"]), flush=True)
    del lib
    torch.cuda.empty_cache()
  del R
  torch.cuda.empty_cache()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--held-tasks", required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--smoke-steps", type=int, default=0)
  parser.add_argument("--lane", default="")
  args = parser.parse_args()
  C.SEED = SEED
  e2 = C.e2
  assert e2.TRAIN_STEPS == 12_800 and e2.ORTHOGONALITY_WEIGHT == 0.02 and e2.GROUP_SPARSITY_WEIGHT == 1e-3
  if args.smoke_steps:
    e2.TRAIN_STEPS = int(args.smoke_steps)
    args.output = args.output.with_name(args.output.name + "_SMOKE")
  args.output.mkdir(parents=True, exist_ok=True)
  path = args.output / "folds.jsonl"
  done = set()
  if path.exists():
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    done = set((int(r["held_task"]), r["setting"]) for r in rows)
  device = args.device
  data = C.load_lang_v5("blind", verify_hash=False)
  episodes = np.load(C.LANG_NPZ, mmap_mode="r")["episode_indices"].astype(np.int64)
  cache = C.DeviceCache(data, device)
  data.state_features = None
  weights = C.channel_weights(int(data.action_chunks.shape[2]), dead_dims=e2.DEAD_ACTION_DIMS, device=device)
  info = dict(protocol=PROTOCOL, seed=SEED, smoke=bool(args.smoke_steps), lane=args.lane,
    environment=C.run_loto.environment(), repo_code_sha256=C.run_loto.code_hashes(), started_unix=time.time())
  (args.output / ("run_info_%d.json" % int(time.time()))).write_text(json.dumps(info, indent=1))
  for held in [int(t) for t in args.held_tasks.split(",") if t.strip()]:
    if not args.smoke_steps and (BASE / "STOP_PEN").exists():
      print("STOP_PEN present", flush=True)
      return
    run_task(held, data, cache, episodes, device, weights, done, path, args.lane)
  print("penalty done", flush=True)


if __name__ == "__main__":
  main()
