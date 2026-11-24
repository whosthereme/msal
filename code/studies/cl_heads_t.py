import argparse
import json
import time
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE)]
import tc_l40
import numpy as np
import torch

from tc_l40 import (K, SHOTS, FEWSHOT_OFFSET, ACTION_K, NOISE_DRAWS, ORACLE_DRAWS, e2, g39_data, held_episode_rows,
  ideal, l40, mean_mse, run_loto, tc_combo, weighted_ls_code, words, DeviceCache, channel_weights)

SEED = 148021
KEEP = ("state_mean", "state_std", "state_encoder.", "action_encoder.", "trunk.", "trunk_gamma.", "trunk_beta.",
  "plain_head.", "plain_mlp.")


def velocity(R, lib, d, z, inputs):
  with torch.no_grad():
    return tc_combo.Adapted(R, lib, "tfilm_nl", d, z).forward_descriptor(inputs["state"], inputs["x"], inputs["t"], None).cpu()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--held", type=int, required=True)
  ap.add_argument("--stored", type=Path, required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--device", default="cuda:0")
  args = ap.parse_args()
  assert e2.TRAIN_STEPS == 12_800
  device, h = args.device, args.held
  args.out.mkdir(parents=True, exist_ok=True)
  data = g39_data.load_g39_data(verify_hash=False)
  cache = DeviceCache(data, device)
  data.state_features = np.empty((0, g39_data.FROZEN_FEATURE_DIM), dtype=np.float16)
  weights = channel_weights(int(data.action_chunks.shape[2]), dead_dims=e2.DEAD_ACTION_DIMS, device=device)
  t0 = time.time()
  fit_ids = [t for t in sorted(l40.INSTRUCTIONS) if t != h]
  R, _ = run_loto.train_arm("l40_tfilm_nl", seed=SEED, fit_ids=fit_ids, data=data, cache=cache, device=device)
  R.eval()
  zero_flow = run_loto.score_forward(R, "l40_tfilm_nl", held=h, seed=SEED, data=data, cache=cache, device=device)
  lib, stage2 = tc_combo.train_library_t(R, seed=SEED, fit_ids=fit_ids, data=data, cache=cache, device=device)

  def scored(d):
    return dict((name, tc_combo.make_blocks_t(R, lib, cache, data.rows_for_task(h, split), d,
      seed=SEED + getattr(run_loto, offset), draws=NOISE_DRAWS)) for name, split, offset in l40.SPLITS)

  val_rows = data.rows_for_task(h, g39_data.SPLIT_VAL)

  def action(d, z):
    return ideal.action_error(tc_combo.Adapted(R, lib, "tfilm_nl", d, z), val_rows, d, cache, SEED, weights,
      NOISE_DRAWS, run_loto.DRAW_STRIDE)

  d_star = torch.tensor(l40.word_descriptor([h]), dtype=torch.float32, device=device)
  zero = torch.zeros(K, dtype=torch.float64, device=device)
  star = scored(d_star)
  rec = dict(score_forward=dict(selection=zero_flow["main_selection"], robustness=zero_flow["main_robustness"]),
    stage2_final_loss=stage2, zero_shot=dict((s, mean_mse(b, zero, weights)) for s, b in star.items()),
    inversion={}, combination={}, action=dict(zero_shot=action(d_star, None)))
  adapt = {}
  for k in SHOTS:
    rows = held_episode_rows(data.episode_indices, data, h, k)
    demo = tc_combo.make_blocks_t(R, lib, cache, rows, d_star, seed=SEED + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
    d_inv = tc_combo.invert_t(R, demo, d_star, weights)
    demo_inv = tc_combo.make_blocks_t(R, lib, cache, rows, d_inv, seed=SEED + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
    z = weighted_ls_code(demo_inv, weights)
    inv = scored(d_inv)
    rec["inversion"][str(k)] = dict((s, mean_mse(b, zero, weights)) for s, b in inv.items())
    rec["combination"][str(k)] = dict((s, mean_mse(b, z, weights)) for s, b in inv.items())
    rec["combination"][str(k)].update(code_norm=float(z.norm()), descriptor_shift=float((d_inv - d_star).norm()),
      demo_rows=int(len(rows)))
    if k in ACTION_K:
      rec["action"]["inversion_%d" % k] = action(d_inv, None)
      rec["action"]["combination_%d" % k] = action(d_inv, z)
      adapt[k] = dict(d=d_inv, z=z)
  stored = None
  for line in args.stored.read_text().splitlines():
    r = json.loads(line) if line.strip() else None
    if r and r["seed"] == SEED and r["arm"] == "tfilm_nl" and r["held_task"] == h:
      stored = r
  anchors = dict((key, stored is not None and rec[key] == stored[key]) for key in rec)
  g = torch.Generator(device="cpu").manual_seed(1234)
  rows8 = data.rows_for_task(h, g39_data.SPLIT_TRAIN)[:8]
  inputs = dict(state=cache.state_features[torch.as_tensor(rows8, device=device)].float(),
    x=torch.randn(8, 16, 7, generator=g).to(device), t=torch.rand(8, generator=g).to(device))
  art = dict(kind="held_trunk", held=h, seed=SEED, instruction=l40.INSTRUCTIONS[h],
    head=dict((k, v.detach().cpu().clone()) for k, v in R.state_dict().items() if k.startswith(KEEP)),
    library=dict((k, v.detach().cpu().clone()) for k, v in lib.atom_out.state_dict().items()),
    d_star=d_star.cpu(), anchors=anchors, offline=rec,
    probe=dict(state=inputs["state"].cpu(), x=inputs["x"].cpu(), t=inputs["t"].cpu(),
      zero_shot_v=velocity(R, lib, d_star, None, inputs)),
    adapt=dict((k, dict(d=v["d"].cpu(), z=v["z"].cpu(), probe_v=velocity(R, lib, v["d"], v["z"], inputs)))
      for k, v in adapt.items()))
  art["seconds"] = time.time() - t0
  torch.save(art, args.out / ("held%d_trunk.pt" % h))
  print("[cl_heads_t] held %d saved (%.0fs) anchors=%s" % (h, art["seconds"], anchors), flush=True)


if __name__ == "__main__":
  main()
