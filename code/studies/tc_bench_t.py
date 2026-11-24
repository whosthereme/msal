import argparse
import json
import statistics
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE)]
import tc_combo
import numpy as np
import torch

from msal_ext import bench_cpu, hyb, run_nl
from msal_ext.hyb import e2, held_episode_rows, run_loto
from run_e35_1_zero_shot_code_loto import ORACLE_DRAWS, weighted_ls_code

SEED, SHOTS, ARMS = 148021, (1, 4, 16), ("film_nl", "tfilm_nl")


def prep(out):
  from msal_exp.data_lang import LANG_NPZ, load_lang_v5
  from lib.frozen_data import DeviceCache
  from lib.pool_v5 import source_task_ids
  from lib.structured_field import channel_weights
  assert e2.TRAIN_STEPS == 12_800
  device = "cuda"
  data = load_lang_v5("blind", verify_hash=True)
  episodes = np.load(LANG_NPZ, mmap_mode="r")["episode_indices"].astype(np.int64)
  cache = DeviceCache(data, device)
  all_ids = [int(t) for t in source_task_ids()]
  held = all_ids[0]
  fit_ids = [t for t in all_ids if t != held]
  out.mkdir(parents=True, exist_ok=True)
  info = dict(held=held, seed=SEED)
  for arm in ARMS:
    R, _ = run_loto.train_arm(arm, seed=SEED, fit_ids=fit_ids, data=data, cache=cache, device=device)
    R.eval()
    train = hyb.train_library if arm == "film_nl" else tc_combo.train_library_t
    lib, _ = train(R, seed=SEED, fit_ids=fit_ids, data=data, cache=cache, device=device)
    torch.save(R.cpu(), out / ("R_%s.pt" % arm))
    torch.save(lib.cpu(), out / ("lib_%s.pt" % arm))
    info["head_parameters_" + arm] = sum(p.numel() for p in R.parameters())
  rows = held_episode_rows(episodes, data, held, max(SHOTS))
  per_k = {k: np.flatnonzero(np.isin(rows, held_episode_rows(episodes, data, held, k))).tolist() for k in SHOTS}
  index = torch.as_tensor(rows, dtype=torch.long, device=device)
  torch.save({"state": cache.state_features[index].cpu(), "actions": cache.action_chunks[index].cpu(),
    "masks": cache.action_valid_masks[index].cpu(), "rows_for_k": per_k,
    "d_star": torch.tensor(run_nl.arms.DESCRIPTORS["saturated"]([held]), dtype=torch.float32),
    "weights": channel_weights(R.config.action_dim, dead_dims=e2.DEAD_ACTION_DIMS, device="cpu"), "held": held},
    out / "rows.pt")
  info["rows_for_k"] = {k: len(v) for k, v in per_k.items()}
  (out / "prep.json").write_text(json.dumps(info, indent=1))
  print(json.dumps(info))


def run(d, arm, k, threads, reps):
  torch.set_num_threads(threads)
  torch.set_num_interop_threads(1)
  R = torch.load(d / ("R_%s.pt" % arm), weights_only=False).eval()
  lib = torch.load(d / ("lib_%s.pt" % arm), weights_only=False).eval()
  blob = torch.load(d / "rows.pt", weights_only=False)
  cache = bench_cpu.RowCache(blob["state"], blob["actions"], blob["masks"])
  rows = np.asarray(blob["rows_for_k"][k])
  d_star, weights = blob["d_star"], blob["weights"]
  seed = SEED + 60_000
  blocks = hyb.make_blocks if arm == "film_nl" else tc_combo.make_blocks_t
  invert = hyb.invert_descriptor if arm == "film_nl" else tc_combo.invert_t

  def msal():
    demo = blocks(R, lib, cache, rows, d_star, seed=seed, draws=ORACLE_DRAWS)
    d_inv = invert(R, demo, d_star, weights)
    demo_inv = blocks(R, lib, cache, rows, d_inv, seed=seed, draws=ORACLE_DRAWS)
    return weighted_ls_code(demo_inv, weights)

  rss0 = bench_cpu.status_kb("VmRSS")
  Path("/proc/self/clear_refs").write_text("5")
  times = []
  for rep in range(reps):
    t0 = time.perf_counter()
    result = msal()
    times.append(time.perf_counter() - t0)
    if rep == 0:
      peak = bench_cpu.status_kb("VmHWM")
    del result
  return dict(arm=arm, method="msal", k=k, threads=threads, rows=int(len(rows)), seconds=times,
    median_seconds=statistics.median(times), rss_after_setup_mb=rss0 / 1024, peak_extra_mb=(peak - rss0) / 1024,
    torch=torch.__version__)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("mode", choices=("prep", "run"))
  parser.add_argument("--dir", type=Path, required=True)
  parser.add_argument("--arm", choices=ARMS)
  parser.add_argument("--k", type=int, default=16)
  parser.add_argument("--threads", type=int, default=4)
  parser.add_argument("--reps", type=int, default=3)
  args = parser.parse_args()
  if args.mode == "prep":
    prep(args.dir)
    return
  if torch.cuda.is_available():
    raise RuntimeError("run must be CPU-only: set CUDA_VISIBLE_DEVICES=")
  res = run(args.dir, args.arm, args.k, args.threads, args.reps)
  with (args.dir / "bench_t.jsonl").open("a") as handle:
    handle.write(json.dumps(res) + "\n")
  print(json.dumps(res), flush=True)


if __name__ == "__main__":
  main()
