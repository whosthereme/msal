import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE)]
import tc_lodo_conf
import tc_lodo_adapt as A
import numpy as np
import torch

PROTOCOL = "review_20261115_loto_coord_v1"
ARMS_RUN = ("film_nl", "lodo_xy6r2")
A.arms.DESCRIPTORS.setdefault("film_nl", A.arms.DESCRIPTORS["saturated"])


def run_unit(arm, held, seed, data, cache, episodes, device, weights, all_ids, direction_of):
  started = time.time()
  fit_ids = [t for t in all_ids if t != held]
  R, _ = A.run_loto.train_arm(arm, seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
  R.eval()
  flow = A.run_loto.score_forward(R, arm, held=held, seed=seed, data=data, cache=cache, device=device)
  lib, stage2 = A.hyb.train_library(R, seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
  r = dict(arm=arm, seed=seed, held_task=held, held_direction=direction_of[held], n_fit=len(fit_ids),
    score_forward=dict(selection=flow["main_selection"], robustness=flow["main_robustness"]), stage2_final_loss=stage2)
  r.update(A.adapt_task(R, lib, arm, held, seed, data, cache, episodes, device, weights))
  r["seconds"] = time.time() - started
  del R, lib
  torch.cuda.empty_cache()
  return r


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--seed", type=int, required=True)
  ap.add_argument("--held-tasks", required=True)
  ap.add_argument("--output", type=Path, required=True)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--smoke-steps", type=int, default=0)
  ap.add_argument("--lane", default="")
  args = ap.parse_args()
  assert A.e2.TRAIN_STEPS == 12_800
  if args.smoke_steps:
    A.e2.TRAIN_STEPS = int(args.smoke_steps)
    args.output = args.output.with_name(args.output.name + "_SMOKE")
  args.output.mkdir(parents=True, exist_ok=True)
  path = args.output / "folds.jsonl"
  done = set()
  if path.exists():
    done = set((int(r["seed"]), r["arm"], int(r["held_task"])) for r in map(json.loads, filter(str.strip, path.read_text().splitlines())))
  device = args.device
  data = A.load_lang_v5("blind", verify_hash=False)
  episodes = np.load(A.LANG_NPZ, mmap_mode="r")["episode_indices"].astype(np.int64)
  cache = A.DeviceCache(data, device)
  data.state_features = None
  weights = A.channel_weights(int(data.action_chunks.shape[2]), dead_dims=A.e2.DEAD_ACTION_DIMS, device=device)
  all_ids = [int(t) for t in A.source_task_ids()]
  direction_of = dict((t.task_id, t.direction_16) for t in A.build_pool())
  info = dict(protocol=PROTOCOL, seed=args.seed, smoke=bool(args.smoke_steps), lane=args.lane,
    environment=A.run_loto.environment(), repo_code_sha256=A.run_loto.code_hashes(), started_unix=time.time())
  (args.output / ("run_info_%d.json" % int(time.time()))).write_text(json.dumps(info, indent=1))
  for held in [int(t) for t in args.held_tasks.split(",") if t.strip()]:
    for arm in ARMS_RUN:
      if (args.seed, arm, held) in done:
        continue
      if not args.smoke_steps and (BASE / "STOP_LOTO_COORD").exists():
        print("STOP_LOTO_COORD present", flush=True)
        return
      try:
        r = run_unit(arm, held, args.seed, data, cache, episodes, device, weights, all_ids, direction_of)
      except FloatingPointError:
        r = run_unit(arm, held, args.seed, data, cache, episodes, device, weights, all_ids, direction_of)
        r["retrained_after_nan"] = True
      r["lane"] = args.lane
      with path.open("a") as handle:
        handle.write(json.dumps(r) + "\n")
      print("[loto-coord %s] seed=%d held=%2d %.0fs" % (arm, args.seed, held, r["seconds"]), flush=True)
  print("loto-coord done", flush=True)


if __name__ == "__main__":
  main()
