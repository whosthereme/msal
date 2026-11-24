import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_REPO = Path(sys.argv[sys.argv.index("--repo") + 1] if "--repo" in sys.argv else
  "/home/ubuntu/work/manuscript6/repo/sal_v2_identifiable")
sys.path[:0] = [str(HERE), str(_REPO), str(_REPO.parent)]
import numpy as np
import torch

import lc_tasks
from msal_ext import run_nl
from msal_ext.cl_heads import HostFeatureCache, probe, velocity
from msal_ext.hyb import (FEWSHOT_OFFSET, K, e2, held_episode_rows, invert_descriptor, make_blocks, mean_mse,
  run_loto, train_library)
from msal_ext.nl import FiLMNonlinearReadout, PlainNonlinearReadout
from lib import g39_data
from lib.structured_field import channel_weights
from run_e35_1_zero_shot_code_loto import NOISE_DRAWS, ORACLE_DRAWS, weighted_ls_code

PROTOCOL = "review_20261116_cl_coord_v1"
SHARDS = HERE / "shards"
TASKS = {1000 + t["task_id"]: t for t in lc_tasks.tasks()}
ARMS_RUN = ("lc_xy6", "lc_h9", "lc_plain")
SPLITS = (("selection", g39_data.SPLIT_VAL, "SELECTION_OFFSET"), ("robustness", g39_data.SPLIT_TEST, "ROBUSTNESS_OFFSET"))
assert e2.DEAD_ACTION_DIMS == ()


def angle_radius(ids):
  t = np.array([TASKS[int(i)]["direction_16"] for i in ids], dtype=np.float64) * 2.0 * math.pi / 16.0
  r = np.array([TASKS[int(i)]["radius_index"] for i in ids], dtype=np.float64) - 1.0
  return t, r


def xy6(ids):
  t, r = angle_radius(ids)
  return np.stack([np.ones_like(t), np.cos(t), np.sin(t), r, r * np.cos(t), r * np.sin(t)], axis=1)


def h9(ids):
  t, r = angle_radius(ids)
  return np.stack([np.ones_like(t), np.cos(t), np.sin(t), np.cos(2 * t), np.sin(2 * t), r, r ** 2, r * np.cos(t),
    r * np.sin(t)], axis=1)


for name, fn in (("lc_xy6", xy6), ("lc_h9", h9)):
  run_nl.arms.DESCRIPTORS[name] = fn
  run_nl.arms.ARMS[name] = dict(family="film", descriptor=name, reg=False)
  run_nl.BUILDERS[name] = (lambda f: (lambda config, ids: FiLMNonlinearReadout(config, ids, f)))(fn)
  run_nl.CONDITIONED[name] = ("film_gamma", "film_beta")
run_nl.arms.ARMS["lc_plain"] = dict(family="plain", descriptor=None, reg=False)
run_nl.BUILDERS["lc_plain"] = lambda config, ids: PlainNonlinearReadout(config, ids)
run_nl.CONDITIONED["lc_plain"] = ()


def held_ids(fold):
  return sorted(i for i, t in TASKS.items() if t["direction_16"] % 4 == fold)


def seen_ref_ids(fold):
  return sorted(i for i, t in TASKS.items() if t["direction_16"] in ((fold + 2) % 16, (fold + 10) % 16))


def load_data(shards=SHARDS, allow_missing=False):
  if allow_missing:
    for tid in [t for t in TASKS if not (shards / ("task%d.npz" % t)).exists()]:
      del TASKS[tid]
  parts = {k: [] for k in ("state_features", "action_chunks", "action_valid_masks", "task_ids", "episode_indices",
    "step_indices", "split_codes")}
  for tid in sorted(TASKS):
    z = np.load(shards / ("task%d.npz" % tid))
    for k in parts:
      parts[k].append(z[k])
  m = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
  assert m["state_features"].shape[1] == g39_data.FROZEN_FEATURE_DIM
  for tid in TASKS:
    for split in (0, 1, 2):
      assert ((m["task_ids"] == tid) & (m["split_codes"] == split)).any(), (tid, split)
  return g39_data.G39Data(state_features=m["state_features"], action_chunks=m["action_chunks"].astype(np.float32),
    action_valid_masks=m["action_valid_masks"].astype(np.float32), task_ids=m["task_ids"].astype(np.int64),
    episode_indices=m["episode_indices"].astype(np.int64), step_indices=m["step_indices"].astype(np.int64),
    split_codes=m["split_codes"].astype(np.int64), sha256="")


def state_dict(model):
  keep = ("state_mean", "state_std", "state_encoder.", "action_encoder.", "trunk.", "film_gamma.", "film_beta.",
    "film_head.", "film_mlp.", "plain_head.", "plain_mlp.")
  return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k.startswith(keep)}


def run(arm, fold, seed, data, cache, device, out):
  t0 = time.time()
  fit_ids = sorted(set(TASKS) - set(held_ids(fold)))
  R, _ = run_loto.train_arm(arm, seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
  R.eval()
  film = arm != "lc_plain"
  weights = channel_weights(R.config.action_dim, dead_dims=e2.DEAD_ACTION_DIMS, device=device)
  inputs = probe(R, None, data, data.rows_for_task(fit_ids[0], g39_data.SPLIT_TRAIN), device)
  art = dict(kind="lc", arm=arm, fold=fold, seed=seed, protocol=PROTOCOL, head=state_dict(R), probe=inputs,
    descriptors={}, adapt={}, probe_v={})
  records = []
  lib = None
  if film:
    lib, stage2 = train_library(R, seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
    art["library"] = {k: v.detach().cpu().clone() for k, v in lib.atom_out.state_dict().items()}
    art["stage2_final_loss"] = stage2
  zero = torch.zeros(K, dtype=torch.float64, device=device)
  for h in held_ids(fold) + seen_ref_ids(fold):
    is_held = h in held_ids(fold)
    flow = run_loto.score_forward(R, arm, held=h, seed=seed, data=data, cache=cache, device=device)
    rec = dict(arm=arm, fold=fold, seed=seed, task=h, held=is_held, zero_shot=dict(selection=flow["main_selection"],
      robustness=flow["main_robustness"]))
    if film:
      d_star = torch.tensor(run_nl.arms.DESCRIPTORS[arm]([h]), dtype=torch.float32, device=device)
      art["descriptors"][h] = d_star.cpu()
      art["probe_v"][h] = velocity(R, None, d_star, None, inputs, device)
      if is_held:
        rows = held_episode_rows(data.episode_indices, data, h, 1)
        demo = make_blocks(R, lib, cache, rows, d_star, seed=seed + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
        d_inv = invert_descriptor(R, demo, d_star, weights)
        demo_inv = make_blocks(R, lib, cache, rows, d_inv, seed=seed + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
        z = weighted_ls_code(demo_inv, weights)
        sc = dict((name, make_blocks(R, lib, cache, data.rows_for_task(h, split), d_inv,
          seed=seed + getattr(run_loto, offset), draws=NOISE_DRAWS)) for name, split, offset in SPLITS)
        rec["msal_k1"] = dict((s, mean_mse(b, z, weights)) for s, b in sc.items())
        rec["msal_k1"].update(code_norm=float(z.norm()), descriptor_shift=float((d_inv - d_star).norm()),
          demo_rows=int(len(rows)))
        art["adapt"][h] = dict(d=d_inv.cpu(), z=z.cpu(), probe_v=velocity(R, lib, d_inv, z, inputs, device))
    records.append(rec)
  if not film:
    with torch.no_grad():
      s, x, t = (inputs[k].to(device) for k in ("state", "x", "t"))
      art["probe_v_plain"] = R.forward_descriptor(s, x, t).cpu()
  art["seconds"] = time.time() - t0
  torch.save(art, out / ("%s_f%d_s%d.pt" % (arm, fold, seed)))
  with (out / "offline.jsonl").open("a") as f:
    for r in records:
      f.write(json.dumps(r) + "\n")
  print("[lc_heads] %s fold %d seed %d saved (%.0fs)" % (arm, fold, seed, art["seconds"]), flush=True)
  del R, lib
  torch.cuda.empty_cache()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--fold", type=int, required=True)
  ap.add_argument("--seed", type=int, required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--arms", default=",".join(ARMS_RUN))
  ap.add_argument("--device", default="cuda")
  ap.add_argument("--smoke-steps", type=int, default=0)
  ap.add_argument("--repo", default=str(_REPO))
  ap.add_argument("--shards", type=Path, default=SHARDS)
  args = ap.parse_args()
  assert e2.TRAIN_STEPS == 12_800
  if args.smoke_steps:
    e2.TRAIN_STEPS = args.smoke_steps
    args.out = args.out.with_name(args.out.name + "_SMOKE")
  args.out.mkdir(parents=True, exist_ok=True)
  data = load_data(args.shards, allow_missing=bool(args.smoke_steps))
  cache = HostFeatureCache(data, args.device)
  (args.out / ("run_info_f%d_s%d_%d.json" % (args.fold, args.seed, int(time.time())))).write_text(json.dumps(dict(
    protocol=PROTOCOL, fold=args.fold, seed=args.seed, rows=int(len(data.task_ids)), smoke=bool(args.smoke_steps),
    held=held_ids(args.fold), seen_ref=seen_ref_ids(args.fold), train_steps=int(e2.TRAIN_STEPS)), indent=1))
  for arm in args.arms.split(","):
    if (args.out / ("%s_f%d_s%d.pt" % (arm, args.fold, args.seed))).exists():
      continue
    run(arm, args.fold, args.seed, data, cache, args.device, args.out)


if __name__ == "__main__":
  main()
