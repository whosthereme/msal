import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "l40")]
import lc_heads2 as h2
import numpy as np
import torch

import tc_combo
from msal_ext.hyb import FEWSHOT_OFFSET, K, held_episode_rows, mean_mse
from run_e35_1_zero_shot_code_loto import NOISE_DRAWS, ORACLE_DRAWS, weighted_ls_code

PROTOCOL = "trunk2_20261121_cl_v1"
ARM_H, ARM_K = "lc_tfilm_h20", "lc_tfilm_xy6"
e2, run_loto, g39_data, lc_heads = h2.e2, h2.run_loto, h2.g39_data, h2.lc_heads
assert e2.DEAD_ACTION_DIMS == ()


class TrunkFiLMFn(h2.tc_arms.TrunkFiLMField):

  def __init__(self, config, source_task_ids, fn):
    h2.PlainNonlinearReadout.__init__(self, config, source_task_ids)
    matrix = fn(source_task_ids)
    self.register_buffer("task_features", torch.tensor(matrix, dtype=torch.float32), persistent=True)
    self.trunk_gamma = h2.tc_arms.bias_free(int(matrix.shape[1]), config.hidden_dim)
    self.trunk_beta = h2.tc_arms.bias_free(int(matrix.shape[1]), config.hidden_dim)


for _arm, _fn in ((ARM_H, h2.h20), (ARM_K, h2.xy6)):
  h2.DESC[_arm] = _fn
  h2.run_nl.arms.DESCRIPTORS[_arm] = _fn
  h2.run_nl.arms.ARMS[_arm] = dict(family="film", descriptor=_arm, reg=False)
  h2.run_nl.BUILDERS[_arm] = (lambda f: lambda c, ids: TrunkFiLMFn(c, ids, f))(_fn)
  h2.run_nl.CONDITIONED[_arm] = ("trunk_gamma", "trunk_beta")


def adapt(fold, seed, data, cache, device, out, stored):
  t0 = time.time()
  art3 = torch.load(stored / ("%s_f%d_s%d.pt" % (ARM_K, fold, seed)), weights_only=False)
  held, seen = h2.held_ids(fold), lc_heads.seen_ref_ids(fold)
  fit_ids = sorted(set(h2.TASKS) - set(held))
  torch.manual_seed(seed)
  R = run_loto.build_model(ARM_K, run_loto.config_for(ARM_K, data), fit_ids).to(device)
  result = R.load_state_dict({k: v.to(device) for k, v in art3["head"].items()}, strict=False)
  assert list(result.missing_keys) == ["task_features"] and not result.unexpected_keys, result
  R.eval()
  inputs = art3["probe"]
  weights = h2.channel_weights(R.config.action_dim, dead_dims=e2.DEAD_ACTION_DIMS, device=device)

  def desc(h):
    return torch.tensor(h2.xy6([h]), dtype=torch.float32, device=device)

  worst = max(float((h2.velocity(R, None, desc(h), None, inputs, device) - art3["probe_v"][h]).abs().max()) for h in held)
  if worst > 1e-5:
    raise RuntimeError("loaded head disagrees with the stored probe (%.2e)" % worst)
  lib, stage2 = tc_combo.train_library_t(R, seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
  art = dict(kind="lc4", arm=ARM_K, fold=fold, seed=seed, protocol=PROTOCOL, head=art3["head"], probe=inputs,
    library={k: v.detach().cpu().clone() for k, v in lib.atom_out.state_dict().items()}, stage2_final_loss=stage2,
    stored_probe_max_abs=worst, descriptors={}, probe_v={}, adapt={})
  zero = torch.zeros(K, dtype=torch.float64, device=device)
  records = []
  for h in held + seen:
    is_held = h in held
    d_star = desc(h)
    flow = run_loto.score_forward(R, ARM_K, held=h, seed=seed, data=data, cache=cache, device=device)
    rec = dict(arm=ARM_K, fold=fold, seed=seed, task=h, held=is_held, zero_shot=dict(selection=flow["main_selection"],
      robustness=flow["main_robustness"]))
    art["descriptors"][h] = d_star.cpu()
    art["probe_v"][h] = h2.velocity(R, None, d_star, None, inputs, device)
    if is_held:
      val_rows = data.rows_for_task(h, g39_data.SPLIT_VAL)

      def act(d, z):
        return h2.ideal.action_error(tc_combo.Adapted(R, lib, "tfilm_nl", d, z), val_rows, d, cache, seed, weights,
          NOISE_DRAWS, run_loto.DRAW_STRIDE)

      rows = held_episode_rows(data.episode_indices, data, h, 1)
      demo = tc_combo.make_blocks_t(R, lib, cache, rows, d_star, seed=seed + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
      d_inv = tc_combo.invert_t(R, demo, d_star, weights)
      demo_inv = tc_combo.make_blocks_t(R, lib, cache, rows, d_inv, seed=seed + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
      z = weighted_ls_code(demo_inv, weights)
      sc = dict((name, tc_combo.make_blocks_t(R, lib, cache, data.rows_for_task(h, split), d_inv,
        seed=seed + getattr(run_loto, offset), draws=NOISE_DRAWS)) for name, split, offset in lc_heads.SPLITS)
      rec["inversion_k1"] = dict((s, mean_mse(b, zero, weights)) for s, b in sc.items())
      rec["msal_k1"] = dict((s, mean_mse(b, z, weights)) for s, b in sc.items())
      rec["msal_k1"].update(code_norm=float(z.norm()), descriptor_shift=float((d_inv - d_star).norm()),
        demo_rows=int(len(rows)))
      rec["action"] = dict(zero_shot=act(d_star, None), msal_k1=act(d_inv, z))
      with torch.no_grad():
        s, x, t = (inputs[k].to(device) for k in ("state", "x", "t"))
        v = tc_combo.Adapted(R, lib, "tfilm_nl", d_inv, z).forward_descriptor(s, x, t, None).cpu()
      art["adapt"][h] = dict(d=d_inv.cpu(), z=z.cpu(), probe_v=v)
    records.append(rec)
  art["seconds"] = time.time() - t0
  torch.save(art, out / ("%sa_f%d_s%d.pt" % (ARM_K, fold, seed)))
  with (out / "offline4.jsonl").open("a") as f:
    for r in records:
      f.write(json.dumps(r) + "\n")
  print("[lc_heads4] adapt fold %d seed %d saved (%.0fs, stored probe %.1e)" % (fold, seed, art["seconds"], worst), flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("mode", choices=("harmonic", "adapt"))
  ap.add_argument("--fold", type=int, required=True)
  ap.add_argument("--seed", type=int, required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--stored", type=Path, default=HERE / "heads3")
  ap.add_argument("--device", default="cuda")
  ap.add_argument("--smoke-steps", type=int, default=0)
  args = ap.parse_args()
  assert e2.TRAIN_STEPS == 12_800
  if args.smoke_steps:
    e2.TRAIN_STEPS = args.smoke_steps
    args.out = args.out.with_name(args.out.name + "_SMOKE")
  args.out.mkdir(parents=True, exist_ok=True)
  name = "%s_f%d_s%d.pt" % (ARM_H if args.mode == "harmonic" else ARM_K + "a", args.fold, args.seed)
  if (args.out / name).exists():
    return
  data = lc_heads.load_data(lc_heads.SHARDS)
  cache = h2.HostFeatureCache(data, args.device)
  (args.out / ("run_info4_%s_f%d_s%d_%d.json" % (args.mode, args.fold, args.seed, int(time.time())))).write_text(
    json.dumps(dict(protocol=PROTOCOL, mode=args.mode, fold=args.fold, seed=args.seed, rows=int(len(data.task_ids)),
      smoke=bool(args.smoke_steps), held=h2.held_ids(args.fold), train_steps=int(e2.TRAIN_STEPS)), indent=1))
  if args.mode == "harmonic":
    h2.PROTOCOL = PROTOCOL
    h2.run(ARM_H, args.fold, args.seed, data, cache, args.device, args.out)
  else:
    adapt(args.fold, args.seed, data, cache, args.device, args.out, args.stored)


if __name__ == "__main__":
  main()
