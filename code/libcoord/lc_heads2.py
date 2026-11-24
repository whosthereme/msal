import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_TC = Path(sys.argv[sys.argv.index("--tc") + 1]) if "--tc" in sys.argv else HERE.parent
sys.path[:0] = [str(HERE), str(_TC)]
import lc_heads
import numpy as np
import torch

import tc_arms
from msal_exp.arms import FiLMField
from msal_ext import ideal, run_nl
from msal_ext.cl_heads import HostFeatureCache, probe, velocity
from msal_ext.hyb import e2, run_loto
from msal_ext.nl import DescriptorMoE, FiLMNonlinearReadout, PlainNonlinearReadout
from lib import g39_data
from lib.structured_field import channel_weights
from run_e35_1_zero_shot_code_loto import NOISE_DRAWS

PROTOCOL = "review_20261119_cl_heads_v1"
ARMS_RUN = ("lc_film_xy6", "lc_moe_xy6", "lc_tin_xy6", "lc_h20")
TASKS, xy6, held_ids = lc_heads.TASKS, lc_heads.xy6, lc_heads.held_ids
assert e2.DEAD_ACTION_DIMS == ()


def h20(ids):
  t, r = lc_heads.angle_radius(ids)
  cols = [np.ones_like(t)] + [np.cos(m * t) for m in range(1, 9)] + [np.sin(m * t) for m in range(1, 8)]
  cols += [r, r ** 2, r * np.cos(t), r * np.sin(t)]
  return np.stack(cols, axis=1)


class TrunkInputXY(tc_arms.TrunkInputField):

  def __init__(self, config, source_task_ids):
    PlainNonlinearReadout.__init__(self, config, source_task_ids)
    matrix = xy6(source_task_ids)
    self.register_buffer("task_features", torch.tensor(matrix, dtype=torch.float32), persistent=True)
    self.desc_in = tc_arms.bias_free(int(matrix.shape[1]), config.hidden_dim)


DESC = dict(lc_film_xy6=xy6, lc_moe_xy6=xy6, lc_tin_xy6=xy6, lc_h20=h20)
BUILD = dict(lc_film_xy6=lambda c, ids: FiLMField(c, ids, xy6), lc_moe_xy6=lambda c, ids: DescriptorMoE(c, ids, xy6),
  lc_tin_xy6=lambda c, ids: TrunkInputXY(c, ids), lc_h20=lambda c, ids: FiLMNonlinearReadout(c, ids, h20))
COND = dict(lc_film_xy6=("film_gamma", "film_beta"), lc_moe_xy6=("atom_out", "gate"), lc_tin_xy6=("desc_in",),
  lc_h20=("film_gamma", "film_beta"))
for _a in ARMS_RUN:
  run_nl.arms.DESCRIPTORS[_a] = DESC[_a]
  run_nl.arms.ARMS[_a] = dict(family="film", descriptor=_a, reg=False)
  run_nl.BUILDERS[_a] = BUILD[_a]
  run_nl.CONDITIONED[_a] = COND[_a]


def span_share(fit, row):
  _, s, vt = np.linalg.svd(np.asarray(fit, dtype=np.float64), full_matrices=False)
  basis = vt[:int((s > 1e-9 * s[0]).sum())]
  p = basis.T @ (basis @ row)
  return float(p @ p / (row @ row))


def run(arm, fold, seed, data, cache, device, out):
  t0 = time.time()
  held = held_ids(fold)
  fit_ids = sorted(set(TASKS) - set(held))
  R, _ = run_loto.train_arm(arm, seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
  R.eval()
  weights = channel_weights(R.config.action_dim, dead_dims=e2.DEAD_ACTION_DIMS, device=device)
  inputs = probe(R, None, data, data.rows_for_task(fit_ids[0], g39_data.SPLIT_TRAIN), device)
  art = dict(kind="lc2", arm=arm, fold=fold, seed=seed, protocol=PROTOCOL,
    head={k: v.detach().cpu().clone() for k, v in R.state_dict().items() if k != "task_features"}, probe=inputs,
    descriptors={}, probe_v={})
  fit_matrix = DESC[arm](fit_ids)
  records = []
  for h in held:
    flow = run_loto.score_forward(R, arm, held=h, seed=seed, data=data, cache=cache, device=device)
    d_np = DESC[arm]([h])
    d_star = torch.tensor(d_np, dtype=torch.float32, device=device)
    act = ideal.action_error(R, data.rows_for_task(h, g39_data.SPLIT_VAL), d_star, cache, seed, weights, NOISE_DRAWS,
      run_loto.DRAW_STRIDE)
    art["descriptors"][h] = d_star.cpu()
    art["probe_v"][h] = velocity(R, None, d_star, None, inputs, device)
    records.append(dict(arm=arm, fold=fold, seed=seed, task=h, held=True, zero_shot=dict(
      selection=flow["main_selection"], robustness=flow["main_robustness"]), action_selection=act,
      iota=span_share(fit_matrix, d_np[0])))
  art["seconds"] = time.time() - t0
  torch.save(art, out / ("%s_f%d_s%d.pt" % (arm, fold, seed)))
  with (out / "offline2.jsonl").open("a") as f:
    for r in records:
      f.write(json.dumps(r) + "\n")
  print("[lc_heads2] %s fold %d seed %d saved (%.0fs)" % (arm, fold, seed, art["seconds"]), flush=True)
  del R
  torch.cuda.empty_cache()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--fold", type=int, required=True)
  ap.add_argument("--seed", type=int, required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--arms", default=",".join(ARMS_RUN))
  ap.add_argument("--device", default="cuda")
  ap.add_argument("--smoke-steps", type=int, default=0)
  ap.add_argument("--repo", default="")
  ap.add_argument("--tc", default="")
  ap.add_argument("--shards", type=Path, default=lc_heads.SHARDS)
  args = ap.parse_args()
  assert e2.TRAIN_STEPS == 12_800
  if args.smoke_steps:
    e2.TRAIN_STEPS = args.smoke_steps
    args.out = args.out.with_name(args.out.name + "_SMOKE")
  args.out.mkdir(parents=True, exist_ok=True)
  data = lc_heads.load_data(args.shards)
  cache = HostFeatureCache(data, args.device)
  (args.out / ("run_info2_f%d_s%d_%d.json" % (args.fold, args.seed, int(time.time())))).write_text(json.dumps(dict(
    protocol=PROTOCOL, fold=args.fold, seed=args.seed, rows=int(len(data.task_ids)), smoke=bool(args.smoke_steps),
    held=held_ids(args.fold), train_steps=int(e2.TRAIN_STEPS)), indent=1))
  for arm in args.arms.split(","):
    if (args.out / ("%s_f%d_s%d.pt" % (arm, args.fold, args.seed))).exists():
      continue
    run(arm, args.fold, args.seed, data, cache, args.device, args.out)


if __name__ == "__main__":
  main()
