import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE), str(BASE / "trunkcond"), str(BASE / "repo" / "sal_v2_identifiable"), str(BASE / "repo")]
if sys.platform == "win32":
  import boot
import numpy as np
import torch

import tc_arms
from msal_exp.arms import _trunk_hidden
from msal_ext import hyb, ideal, run_nl
from msal_ext.hyb import (K, SHOTS, FEWSHOT_OFFSET, e2, held_episode_rows, mean_mse, run_loto,
  output_orthogonality_error, group_sparsity_penalty)
from msal_exp.data_lang import LANG_NPZ, SPLIT_ROBUSTNESS, SPLIT_SELECTION, SPLIT_TRAIN, load_lang_v5
from lib.frozen_data import DeviceCache
from lib.pool_v5 import source_task_ids
from lib.structured_field import channel_weights
from run_e35_1_zero_shot_code_loto import NOISE_DRAWS, ORACLE_DRAWS, weighted_ls_code

PROTOCOL = "review_20261113_adapt_trunk_v1"
SEED = 148021
ARMS = ("film_nl", "tfilm_nl")
SPLITS = (("selection", SPLIT_SELECTION, "SELECTION_OFFSET"), ("robustness", SPLIT_ROBUSTNESS, "ROBUSTNESS_OFFSET"))
ACTION_K = (1, 16)


def hidden_at(R, arm, state, x, t, d):
  if arm == "film_nl":
    return _trunk_hidden(R, state, x, t)
  first = R.trunk[1](R.trunk[0](tc_arms.trunk_inputs(R, state, x, t)))
  first = first * (1.0 + R.trunk_gamma(d)) + R.trunk_beta(d)
  return tc_arms.trunk_tail(R, first)


def train_library_t(R, *, seed, fit_ids, data, cache, device):
  R.eval()
  for p in R.parameters():
    p.requires_grad_(False)
  lib = hyb.new_library(R, seed, len(fit_ids), device)
  lookup = dict((t, i) for i, t in enumerate(fit_ids))
  weights = channel_weights(R.config.action_dim, dead_dims=e2.DEAD_ACTION_DIMS, device=device)
  rows_by_task = dict((t, data.rows_for_task(t, SPLIT_TRAIN)) for t in fit_ids)
  optimizer = torch.optim.AdamW(lib.parameters(), lr=e2.LEARNING_RATE, weight_decay=e2.WEIGHT_DECAY)
  rng = np.random.default_rng(seed + hyb.STAGE2_RNG_OFFSET)
  per_task = max(e2.BATCH_SIZE // len(fit_ids), 1)
  loss = torch.tensor(float("nan"))
  for step in range(1, e2.TRAIN_STEPS + 1):
    picked = np.concatenate([rng.choice(rows_by_task[t], size=per_task, replace=True) for t in fit_ids])
    batch = cache.batch(picked, seed=seed * 7919 + step + hyb.STAGE2_BATCH_OFFSET)
    task_rows = torch.tensor([lookup[int(v)] for v in data.task_ids[picked]], device=device)
    with torch.no_grad():
      base = R.forward_with_rows(batch["state"], batch["x_t"], batch["time"], task_rows)
      d = R.task_features[task_rows.long()]
      hidden = hidden_at(R, "tfilm_nl", batch["state"], batch["x_t"], batch["time"], d)
    atoms = lib.fields(hidden)
    prediction = base + torch.einsum("bk,bktd->btd", lib.codes[task_rows], atoms)
    loss = (e2.masked_channel_loss(prediction, batch["target"], batch["masks"], weights)
      + e2.ORTHOGONALITY_WEIGHT * output_orthogonality_error(atoms)
      + e2.GROUP_SPARSITY_WEIGHT * group_sparsity_penalty(lib.codes, atoms))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
  if not torch.isfinite(loss):
    raise FloatingPointError("stage-2 loss is not finite")
  lib.eval()
  return lib, float(loss.detach())


def make_blocks_t(R, lib, cache, rows, descriptor, *, seed, draws):
  blocks = []
  with torch.no_grad():
    for draw in range(draws):
      batch = cache.batch(rows, seed=seed + hyb.DRAW_STRIDE * draw)
      n = batch["state"].shape[0]
      d = descriptor.expand(n, -1)
      base = R.forward_descriptor(batch["state"], batch["x_t"], batch["time"], d)
      atoms = lib.fields(hidden_at(R, "tfilm_nl", batch["state"], batch["x_t"], batch["time"], d))
      blocks.append({"base": base.double(), "atoms": atoms.double(), "target": batch["target"].double(),
        "mask": batch["masks"].double(), "batch": batch})
  return blocks


def invert_t(R, blocks, d0, weights):
  firsts = []
  with torch.no_grad():
    for b in blocks:
      x = b["batch"]
      firsts.append(R.trunk[1](R.trunk[0](tc_arms.trunk_inputs(R, x["state"], x["x_t"], x["time"]))))
  d = d0.clone().requires_grad_(True)
  optimizer = torch.optim.Adam([d], lr=hyb.INVERSION_LR)
  for _ in range(hyb.INVERSION_STEPS):
    loss = 0.0
    for b, first in zip(blocks, firsts):
      n = first.shape[0]
      dd = d.expand(n, -1)
      hidden = tc_arms.trunk_tail(R, first * (1.0 + R.trunk_gamma(dd)) + R.trunk_beta(dd))
      prediction = tc_arms.shape_out(R, R.plain_head(hidden) + R.plain_mlp(hidden), n)
      loss = loss + e2.masked_channel_loss(prediction, b["batch"]["target"], b["batch"]["masks"], weights)
    optimizer.zero_grad(set_to_none=True)
    (loss / len(blocks)).backward()
    optimizer.step()
  return d.detach()


class Adapted:

  def __init__(self, R, lib, arm, d, z):
    self.R, self.lib, self.arm, self.d, self.z = R, lib, arm, d, z

  def forward_descriptor(self, state, x, t, _):
    n = state.shape[0]
    dd = self.d.expand(n, -1)
    out = self.R.forward_descriptor(state, x, t, dd)
    if self.z is not None:
      atoms = self.lib.fields(hidden_at(self.R, self.arm, state, x, t, dd))
      out = out + torch.einsum("k,bktd->btd", self.z.float(), atoms)
    return out


def run_fold(arm, held, data, cache, episodes, device, weights):
  started = time.time()
  fit_ids = [t for t in source_task_ids() if t != held]
  R, _ = run_loto.train_arm(arm, seed=SEED, fit_ids=fit_ids, data=data, cache=cache, device=device)
  R.eval()
  zero_flow = run_loto.score_forward(R, arm, held=held, seed=SEED, data=data, cache=cache, device=device)
  train = hyb.train_library if arm == "film_nl" else train_library_t
  lib, stage2 = train(R, seed=SEED, fit_ids=fit_ids, data=data, cache=cache, device=device)
  blocks = hyb.make_blocks if arm == "film_nl" else make_blocks_t
  invert = hyb.invert_descriptor if arm == "film_nl" else invert_t

  def scored(d):
    return dict((name, blocks(R, lib, cache, data.rows_for_task(held, split), d,
      seed=SEED + getattr(run_loto, offset), draws=NOISE_DRAWS)) for name, split, offset in SPLITS)

  sel_rows = data.rows_for_task(held, SPLIT_SELECTION)

  def action(d, z):
    return ideal.action_error(Adapted(R, lib, arm, d, z), sel_rows, d, cache, SEED, weights, NOISE_DRAWS,
      run_loto.DRAW_STRIDE)

  d_star = torch.tensor(run_nl.arms.DESCRIPTORS["saturated"]([held]), dtype=torch.float32, device=device)
  zero = torch.zeros(K, dtype=torch.float64, device=device)
  star = scored(d_star)
  record = dict(arm=arm, held_task=held, seed=SEED, score_forward_selection=zero_flow["main_selection"],
    stage2_final_loss=stage2, zero_shot=dict((s, mean_mse(b, zero, weights)) for s, b in star.items()),
    inversion={}, combination={}, action=dict(zero_shot=action(d_star, None)))
  for k in SHOTS:
    rows = held_episode_rows(episodes, data, held, k)
    demo = blocks(R, lib, cache, rows, d_star, seed=SEED + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
    d_inv = invert(R, demo, d_star, weights)
    demo_inv = blocks(R, lib, cache, rows, d_inv, seed=SEED + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
    z = weighted_ls_code(demo_inv, weights)
    inv = scored(d_inv)
    record["inversion"][str(k)] = dict((s, mean_mse(b, zero, weights)) for s, b in inv.items())
    record["combination"][str(k)] = dict((s, mean_mse(b, z, weights)) for s, b in inv.items())
    record["combination"][str(k)].update(code_norm=float(z.norm()), descriptor_shift=float((d_inv - d_star).norm()))
    if k in ACTION_K:
      record["action"]["inversion_%d" % k] = action(d_inv, None)
      record["action"]["combination_%d" % k] = action(d_inv, z)
  record["seconds"] = time.time() - started
  del R, lib
  torch.cuda.empty_cache()
  return record


def main():
  parser = argparse.ArgumentParser()
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
    done = set((r["arm"], int(r["held_task"])) for r in map(json.loads, path.read_text().splitlines()) if r)
  device = args.device
  data = load_lang_v5("blind", verify_hash=False)
  episodes = np.load(LANG_NPZ, mmap_mode="r")["episode_indices"].astype(np.int64)
  cache = DeviceCache(data, device)
  data.state_features = None
  weights = channel_weights(int(data.action_chunks.shape[2]), dead_dims=e2.DEAD_ACTION_DIMS, device=device)
  info = dict(protocol=PROTOCOL, seed=SEED, smoke=bool(args.smoke_steps), lane=args.lane,
    environment=run_loto.environment(), repo_code_sha256=run_loto.code_hashes(), started_unix=time.time())
  (args.output / ("run_info_%d.json" % int(time.time()))).write_text(json.dumps(info, indent=1))
  for held in [int(t) for t in args.held_tasks.split(",") if t.strip()]:
    for arm in ARMS:
      if (arm, held) in done:
        continue
      if not args.smoke_steps and (BASE / "STOP_ADAPT").exists():
        print("STOP_ADAPT present", flush=True)
        return
      try:
        record = run_fold(arm, held, data, cache, episodes, device, weights)
      except FloatingPointError:
        record = run_fold(arm, held, data, cache, episodes, device, weights)
        record["retrained_after_nan"] = True
      record["lane"] = args.lane
      with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
      print("[adapt %s] held=%2d %.0fs" % (arm, held, record["seconds"]), flush=True)
  print("adapt done", flush=True)


if __name__ == "__main__":
  main()
