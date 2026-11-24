from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from msal_ext import l40
from msal_ext.hyb import (FEWSHOT_OFFSET, K, e2, held_episode_rows, invert_descriptor, make_blocks, mean_mse,
                            run_loto, train_library, new_library, _trunk_hidden)
from msal_ext.l40 import INSTRUCTIONS, scored, word_descriptor
from lib import g39_data
from lib.frozen_data import DeviceCache
from lib.structured_field import channel_weights
from run_e35_1_zero_shot_code_loto import ORACLE_DRAWS, weighted_ls_code

SEED = 148021
SHOTS = (1, 16)
FT_BATCH, FT_RNG_TAG, FT_BATCH_OFFSET, FT_LR = 512, 7, 70_000_000, 1e-4
FT_STEPS = {1: 100, 2: 100, 4: 400, 8: 400, 16: 400}
RUNS = Path("/home/ubuntu/work/manuscript6/runs")


def head_state(model) -> dict:
    keep = ("state_mean", "state_std", "state_encoder.", "action_encoder.", "trunk.", "film_gamma.", "film_beta.",
            "film_head.", "film_mlp.")
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k.startswith(keep)}


def probe(R, lib, data, rows, device):
    g = torch.Generator(device="cpu").manual_seed(1234)
    state = torch.as_tensor(data.state_features[rows[:8]], dtype=torch.float32)
    x = torch.randn(8, 16, 7, generator=g)
    t = torch.rand(8, generator=g)
    return {"state": state, "x": x, "t": t}


def velocity(R, lib, d, z, inputs, device):
    with torch.no_grad():
        s, x, t = (inputs[k].to(device) for k in ("state", "x", "t"))
        n = s.shape[0]
        v = R.forward_descriptor(s, x, t, d.to(device).expand(n, -1))
        if z is not None:
            atoms = lib.fields(_trunk_hidden(R, s, x, t))
            v = v + torch.einsum("k,bktd->btd", z.to(device).float(), atoms)
    return v.cpu()


class HostFeatureCache(DeviceCache):

    def __init__(self, data, device: str = "cuda") -> None:
        self.device = device
        self.host_features = torch.from_numpy(data.state_features)
        self.action_chunks = torch.from_numpy(data.action_chunks).to(device)
        self.action_valid_masks = torch.from_numpy(data.action_valid_masks).to(device)

    def batch(self, rows, *, seed: int):
        index_host = torch.as_tensor(np.asarray(rows), dtype=torch.long)
        index = index_host.to(self.device)
        state = self.host_features[index_host].to(self.device).float()
        actions = self.action_chunks[index]
        masks = self.action_valid_masks[index]
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        noise = torch.randn(actions.shape, generator=generator, device=self.device)
        t = torch.rand(actions.shape[0], generator=generator, device=self.device) * 0.998 + 0.001
        x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
        return {"state": state, "actions": actions, "masks": masks, "time": t, "x_t": x_t, "target": noise - actions}


def load(device):
    data = g39_data.load_g39_data(verify_hash=True)
    cache = HostFeatureCache(data, device)
    return data, cache


def stored_l40(held):
    for p in sorted((RUNS / "l40").glob(f"seed{SEED}_*/folds.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip() and json.loads(line)["held_task"] == held:
                return json.loads(line)
    return None


def stored_ft(held):
    for p in sorted((RUNS / "ft").glob(f"*_l40_*_s{SEED}/folds.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip() and json.loads(line)["held_task"] == held:
                return json.loads(line)
    return None


def seen(out: Path, device: str) -> None:
    data, cache = load(device)
    fit_ids = sorted(INSTRUCTIONS)
    t0 = time.time()
    R, _ = run_loto.train_arm("l40_film_nl", seed=SEED, fit_ids=fit_ids, data=data, cache=cache, device=device)
    R.eval()
    inputs = probe(R, None, data, data.rows_for_task(fit_ids[0], g39_data.SPLIT_TRAIN), device)
    d0 = torch.tensor(word_descriptor([fit_ids[0]]), dtype=torch.float32)
    torch.save({"kind": "seen", "seed": SEED, "head": head_state(R), "descriptors": {t: torch.tensor(word_descriptor([t]), dtype=torch.float32) for t in fit_ids},
                "probe": {**inputs, "d": d0, "v": velocity(R, None, d0, None, inputs, device)},
                "train_seconds": time.time() - t0}, out / "seen.pt")
    print(f"[cl_heads] seen head saved ({time.time() - t0:.0f}s)", flush=True)


def held(out: Path, h: int, device: str) -> None:
    data, cache = load(device)
    fit_ids = [t for t in sorted(INSTRUCTIONS) if t != h]
    t0 = time.time()
    R, r32 = l40.zero_shot("l40_film_nl", held=h, seed=SEED, data=data, cache=cache, device=device, fit_ids=fit_ids)
    lib, _ = train_library(R, seed=SEED, fit_ids=fit_ids, data=data, cache=cache, device=device)
    weights = channel_weights(R.config.action_dim, dead_dims=e2.DEAD_ACTION_DIMS, device=device)
    d_star = torch.tensor(word_descriptor([h]), dtype=torch.float32, device=device)
    zero = torch.zeros(K, dtype=torch.float64, device=device)
    ref, ref_ft = stored_l40(h), stored_ft(h)
    anchors = {"film_nl_float32": ref is not None and ref["zero_shot_float32"]["film_nl"] == r32}
    star = scored(R, lib, cache, data, h, d_star, SEED)
    anchors["zero_shot"] = ref is not None and ref["zero_shot"] == {s: mean_mse(b, zero, weights) for s, b in star.items()}
    inputs = probe(R, lib, data, data.rows_for_task(h, g39_data.SPLIT_TRAIN), device)
    art = {"kind": "held", "held": h, "seed": SEED, "instruction": INSTRUCTIONS[h], "head": head_state(R),
           "library": {k: v.detach().cpu().clone() for k, v in lib.atom_out.state_dict().items()},
           "d_star": d_star.cpu(), "adapt": {}, "finetune": {}, "anchors": anchors,
           "probe": {**inputs, "zero_shot_v": velocity(R, lib, d_star, None, inputs, device)}}
    for k in SHOTS:
        rows = held_episode_rows(data.episode_indices, data, h, k)
        demo = make_blocks(R, lib, cache, rows, d_star, seed=SEED + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
        d_inv = invert_descriptor(R, demo, d_star, weights)
        demo_inv = make_blocks(R, lib, cache, rows, d_inv, seed=SEED + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
        z = weighted_ls_code(demo_inv, weights)
        inv = scored(R, lib, cache, data, h, d_inv, SEED)
        msal_sel = mean_mse(inv["selection"], z, weights)
        anchors[f"msal_k{k}"] = ref is not None and ref["combination"][str(k)]["selection"] == msal_sel
        art["adapt"][k] = {"d": d_inv.cpu(), "z": z.cpu(), "selection": msal_sel,
                           "probe_v": velocity(R, lib, d_inv, z, inputs, device)}
        model = copy.deepcopy(R)
        for p in model.parameters():
            p.requires_grad_(True)
        optimizer = torch.optim.Adam(list(model.parameters()), lr=FT_LR)
        rng = np.random.default_rng([SEED, h, k, 1, 0, FT_RNG_TAG])
        for step in range(1, FT_STEPS[k] + 1):
            picked = rng.choice(rows, size=FT_BATCH, replace=True)
            batch = cache.batch(picked, seed=SEED * 7919 + step + FT_BATCH_OFFSET + 1_000 * k)
            pred = model.forward_descriptor(batch["state"], batch["x_t"], batch["time"], d_star.expand(len(picked), -1))
            loss = e2.masked_channel_loss(pred, batch["target"], batch["masks"], weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        dummy = new_library(R, SEED, len(fit_ids), device).eval()
        with torch.no_grad():
            ft_sel = mean_mse(scored(model, dummy, cache, data, h, d_star, SEED)["selection"], zero, weights)
        key = f"k{k}_all_lr0.0001_s{FT_STEPS[k]}"
        anchors[f"ft_k{k}"] = ref_ft is not None and ref_ft["finetune"].get(key, {}).get("selection") == ft_sel
        art["finetune"][k] = {"head": head_state(model), "selection": ft_sel, "setting": key,
                              "probe_v": velocity(model, None, d_star, None, inputs, device)}
        del model, optimizer
    art["seconds"] = time.time() - t0
    torch.save(art, out / f"held{h}.pt")
    print(f"[cl_heads] held {h} saved ({art['seconds']:.0f}s) anchors={anchors}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("seen", "held"))
    parser.add_argument("--held", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if e2.TRAIN_STEPS != 12_800:
        raise RuntimeError("unset G12_TRAIN_STEPS")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "seen":
        seen(args.out, args.device)
    else:
        held(args.out, args.held, args.device)


if __name__ == "__main__":
    main()
