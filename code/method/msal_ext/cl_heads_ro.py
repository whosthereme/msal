from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch

from msal_ext import l40
from msal_ext.cl_heads import (FT_BATCH, FT_BATCH_OFFSET, FT_RNG_TAG, RUNS, SEED, head_state, load, probe,
                                 stored_ft, stored_l40, velocity)
from msal_ext.ft import trainable
from msal_ext.hyb import K, e2, held_episode_rows, mean_mse, new_library
from msal_ext.l40 import INSTRUCTIONS, scored, word_descriptor
from lib import g39_data
from lib.structured_field import channel_weights

SETTINGS = ((1, 0, 1e-4, 100), (16, 1, 1e-3, 400))
VARIANT_INDEX = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if e2.TRAIN_STEPS != 12_800:
        raise RuntimeError("unset G12_TRAIN_STEPS")
    args.out.mkdir(parents=True, exist_ok=True)
    h, device = args.held, args.device
    data, cache = load(device)
    fit_ids = [t for t in sorted(INSTRUCTIONS) if t != h]
    t0 = time.time()
    R, r32 = l40.zero_shot("l40_film_nl", held=h, seed=SEED, data=data, cache=cache, device=device, fit_ids=fit_ids)
    weights = channel_weights(R.config.action_dim, dead_dims=e2.DEAD_ACTION_DIMS, device=device)
    d_star = torch.tensor(word_descriptor([h]), dtype=torch.float32, device=device)
    zero = torch.zeros(K, dtype=torch.float64, device=device)
    dummy = new_library(R, SEED, len(fit_ids), device).eval()
    ref, ref_ft = stored_l40(h), stored_ft(h)
    old = torch.load(RUNS / "cl_adapt" / f"held{h}.pt", weights_only=False)
    base = head_state(R)
    anchors = {"film_nl_float32": ref is not None and ref["zero_shot_float32"]["film_nl"] == r32,
               "head_equals_closed_loop_artifact": set(base) == set(old["head"]) and all(
                   torch.equal(base[k], old["head"][k]) for k in base)}
    inputs = probe(R, None, data, data.rows_for_task(h, g39_data.SPLIT_TRAIN), device)
    art = {"kind": "held", "held": h, "seed": SEED, "instruction": INSTRUCTIONS[h], "head": base,
           "d_star": d_star.cpu(), "adapt": {}, "finetune": {}, "anchors": anchors, "variant": "readout",
           "probe": {**inputs, "zero_shot_v": velocity(R, None, d_star, None, inputs, device)}}
    for k, lr_i, lr, steps in SETTINGS:
        rows = held_episode_rows(data.episode_indices, data, h, k)
        model = copy.deepcopy(R)
        for p in model.parameters():
            p.requires_grad_(False)
        params = trainable(model, "readout")
        for p in params:
            p.requires_grad_(True)
        optimizer = torch.optim.Adam(params, lr=lr)
        rng = np.random.default_rng([SEED, h, k, VARIANT_INDEX, lr_i, FT_RNG_TAG])
        for step in range(1, steps + 1):
            picked = rng.choice(rows, size=FT_BATCH, replace=True)
            batch = cache.batch(picked, seed=SEED * 7919 + step + FT_BATCH_OFFSET + 1_000 * k)
            pred = model.forward_descriptor(batch["state"], batch["x_t"], batch["time"], d_star.expand(len(picked), -1))
            loss = e2.masked_channel_loss(pred, batch["target"], batch["masks"], weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"fine-tuning loss not finite (k={k})")
        model.eval()
        with torch.no_grad():
            sel = mean_mse(scored(model, dummy, cache, data, h, d_star, SEED)["selection"], zero, weights)
        key = f"k{k}_readout_lr{lr:g}_s{steps}"
        stored = None if ref_ft is None else ref_ft["finetune"].get(key, {}).get("selection")
        anchors[f"roft_k{k}"] = stored is not None and stored == sel
        new = head_state(model)
        changed = sorted(n for n in new if not torch.equal(new[n], base[n]))
        if not all(n.startswith(("film_head.", "film_mlp.")) for n in changed):
            raise RuntimeError(f"parameters outside the readout changed: {changed}")
        art["finetune"][k] = {"head": new, "selection": sel, "stored_selection": stored, "setting": key,
                              "changed": changed, "probe_v": velocity(model, None, d_star, None, inputs, device)}
        del model, optimizer
    art["seconds"] = time.time() - t0
    torch.save(art, args.out / f"roft{h}.pt")
    print(f"[cl_heads_ro] held {h} saved ({art['seconds']:.0f}s) anchors={anchors} "
          f"selection={ {k: v['selection'] for k, v in art['finetune'].items()} }", flush=True)


if __name__ == "__main__":
    main()
