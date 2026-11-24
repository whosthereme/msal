from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from msal_ext import l40
from msal_ext.cl_heads import SEED, SHOTS, head_state, load, probe, velocity
from msal_ext.hyb import (FEWSHOT_OFFSET, K, _trunk_hidden, e2, held_episode_rows, invert_descriptor, make_blocks,
                            mean_mse, new_library, run_loto)
from msal_ext.l40 import INSTRUCTIONS, word_descriptor
from msal_exp.arms import config_for
from joint_sal_velocity import output_orthogonality_error
from lib import g39_data
from lib.structured_field import channel_weights
from run_e35_1_zero_shot_code_loto import NOISE_DRAWS, ORACLE_DRAWS, weighted_ls_code

RUNS = Path("/home/ubuntu/work/manuscript6/runs/cl_adapt")
META_OFFSET = 80_000_000
QUERY_OFFSET = 10_000_000

CFG = dict(tasks_per_step=8, support_rows=600, query_rows=64, shots=(1, 4, 16), inner_steps=200, inner_lr=1e-2,
           p_corrupt=0.5, word_drop=0.5, zero_shot_weight=1.0, ridge=1e-6, lr=5e-5, weight_decay=1e-5,
           grad_clip=1.0, steps=1500, train="all", query_split="train")


def rebuild(art, data, fit_ids, device):
    config = config_for("l40_film_nl", data)
    torch.manual_seed(SEED)
    R = run_loto.build_model("l40_film_nl", config, fit_ids).to(device)
    res = R.load_state_dict(art["head"], strict=False)
    if res.unexpected_keys or any(not k.startswith("task_features") for k in res.missing_keys):
        raise RuntimeError(f"state dict mismatch: {res}")
    lib = new_library(R, SEED, len(fit_ids), device)
    lib.atom_out.load_state_dict(art["library"])
    return R, lib


def film(R, h, dd):
    n = h.shape[0]
    m = h * (1.0 + R.film_gamma(dd)) + R.film_beta(dd)
    return (R.film_head(m) + R.film_mlp(m)).reshape(n, R.config.chunk_length, R.config.action_dim)


def task_losses(pred, target, mask, weights, idx, n):
    per_row = (((pred - target) ** 2) * weights[None, None, :]).sum(-1) / weights.sum().clamp_min(1e-12)
    num = torch.zeros(n, device=pred.device, dtype=pred.dtype).index_add(0, idx, (per_row * mask).sum(-1))
    den = torch.zeros(n, device=pred.device, dtype=pred.dtype).index_add(0, idx, mask.sum(-1))
    return num / den.clamp_min(1.0)


def task_codes(base, atoms, target, mask, weights, idx, n, ridge):
    scale = weights.sqrt()[None, None, :] * mask.sqrt()[:, :, None]
    A = atoms * scale[:, None]
    r = (target - base) * scale
    G = torch.zeros(n, K, K, device=atoms.device).index_add(0, idx, torch.einsum("nktd,nltd->nkl", A, A))
    b = torch.zeros(n, K, device=atoms.device).index_add(0, idx, torch.einsum("nktd,ntd->nk", A, r))
    G, b = G.double(), b.double()
    lam = ridge * G.diagonal(dim1=1, dim2=2).mean(-1).clamp_min(1e-12)
    eye = torch.eye(K, dtype=torch.float64, device=G.device)
    return torch.linalg.solve(G + lam[:, None, None] * eye, b[..., None])[..., 0].float()


def invert_batch(R, h, target, mask, weights, idx, d0, steps, lr):
    D = d0.clone().requires_grad_(True)
    opt = torch.optim.Adam([D], lr=lr)
    hd = h.detach()
    for _ in range(steps):
        loss = task_losses(film(R, hd, D[idx]), target, mask, weights, idx, D.shape[0]).sum()
        (grad,) = torch.autograd.grad(loss, D)
        D.grad = grad
        opt.step()
    return D.detach()


class Episodes:
    def __init__(self, data, tasks, rng):
        self.tasks, self.rng = list(tasks), rng
        self.rows, self.eps = {}, {}
        for t in self.tasks:
            r = data.rows_for_task(t, g39_data.SPLIT_TRAIN)
            self.rows[t], self.eps[t] = r, data.episode_indices[r]
        self.val = {t: data.rows_for_task(t, g39_data.SPLIT_VAL) for t in self.tasks}
        self.desc = {t: word_descriptor([t])[0] for t in self.tasks}

    def corrupt(self, d, drop):
        d = d.copy()
        words = np.flatnonzero(d[:-1] > 0)
        d[words[self.rng.random(len(words)) < drop]] = 0.0
        return d

    def sample(self, cfg):
        rng = self.rng
        tasks = rng.choice(self.tasks, size=cfg["tasks_per_step"], replace=False)
        sup, qry, sidx, qidx, d0, dt, shots, zer, zidx = [], [], [], [], [], [], [], [], []
        for i, t in enumerate(tasks):
            unique = np.unique(self.eps[t])
            k = int(min(rng.choice(cfg["shots"]), len(unique) - 1))
            chosen = rng.choice(unique, size=k, replace=False)
            inside = np.isin(self.eps[t], chosen)
            s = self.rows[t][inside]
            if len(s) > cfg["support_rows"]:
                s = rng.choice(s, size=cfg["support_rows"], replace=False)
            pool = self.val[t] if cfg["query_split"] == "val" else self.rows[t][~inside]
            q = rng.choice(pool, size=cfg["query_rows"], replace=len(pool) < cfg["query_rows"])
            sup.append(s); qry.append(q)
            sidx.append(np.full(len(s), i)); qidx.append(np.full(len(q), i))
            dt.append(self.desc[t])
            d0.append(self.corrupt(self.desc[t], cfg["word_drop"]) if rng.random() < cfg["p_corrupt"] else self.desc[t])
            shots.append(k)
            zr = rng.choice(self.rows[t], size=cfg["query_rows"], replace=False)
            zer.append(zr); zidx.append(np.full(len(zr), i))
        return (np.concatenate(sup), np.concatenate(qry), np.concatenate(sidx), np.concatenate(qidx),
                np.stack(d0), np.stack(dt), shots, np.concatenate(zer), np.concatenate(zidx))


def meta_train(R, lib, data, cache, tasks, cfg, device, seed_tag, log):
    weights = channel_weights(R.config.action_dim, dead_dims=e2.DEAD_ACTION_DIMS, device=device)
    if cfg["train"] == "all":
        params = list(R.parameters())
    elif cfg["train"] == "film_atoms":
        params = [p for n, p in R.named_parameters() if n.startswith("film_")]
    else:
        params = []
    params += list(lib.atom_out.parameters())
    for p in params:
        p.requires_grad_(True)
    trunk_grad = cfg["train"] == "all"
    use_zero = cfg["train"] != "atoms" and cfg["zero_shot_weight"] > 0
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    episodes = Episodes(data, tasks, np.random.default_rng([SEED, seed_tag, 77]))
    hist, acc, t0 = [], np.zeros(4), time.time()
    for step in range(1, cfg["steps"] + 1):
        srows, qrows, sidx, qidx, d0, dt, _, zrows, zidx = episodes.sample(cfg)
        zidx_t = torch.as_tensor(zidx, device=device)
        n = len(d0)
        sidx_t = torch.as_tensor(sidx, device=device)
        qidx_t = torch.as_tensor(qidx, device=device)
        D0 = torch.as_tensor(d0, dtype=torch.float32, device=device)
        DT = torch.as_tensor(dt, dtype=torch.float32, device=device)
        sb = cache.batch(srows, seed=SEED * 7919 + step + META_OFFSET + 1_000_000 * seed_tag)
        qb = cache.batch(qrows, seed=SEED * 7919 + step + META_OFFSET + QUERY_OFFSET + 1_000_000 * seed_tag)
        with torch.set_grad_enabled(trunk_grad):
            h_s = _trunk_hidden(R, sb["state"], sb["x_t"], sb["time"])
        D = invert_batch(R, h_s, sb["target"], sb["masks"], weights, sidx_t, D0, cfg["inner_steps"], cfg["inner_lr"])
        z = task_codes(film(R, h_s, D[sidx_t]), lib.fields(h_s), sb["target"], sb["masks"], weights, sidx_t, n,
                       cfg["ridge"])
        with torch.set_grad_enabled(trunk_grad):
            h_q = _trunk_hidden(R, qb["state"], qb["x_t"], qb["time"])
        atoms_q = lib.fields(h_q)
        pred = film(R, h_q, D[qidx_t]) + torch.einsum("nk,nktd->ntd", z[qidx_t], atoms_q)
        l_meta = task_losses(pred, qb["target"], qb["masks"], weights, qidx_t, n).mean()
        if use_zero:
            zb = cache.batch(zrows, seed=SEED * 7919 + step + META_OFFSET + 2 * QUERY_OFFSET + 1_000_000 * seed_tag)
            with torch.set_grad_enabled(trunk_grad):
                h_z = _trunk_hidden(R, zb["state"], zb["x_t"], zb["time"])
            l_zero = e2.masked_channel_loss(film(R, h_z, DT[zidx_t]), zb["target"], zb["masks"], weights)
        else:
            l_zero = torch.zeros((), device=device)
        l_orth = output_orthogonality_error(atoms_q)
        loss = l_meta + cfg["zero_shot_weight"] * l_zero + e2.ORTHOGONALITY_WEIGHT * l_orth
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        opt.step()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"meta loss not finite at step {step}")
        acc += [float(l_meta.detach()), float(l_zero.detach()), float(l_orth.detach()), float(z.detach().norm(dim=1).mean())]
        if step % 100 == 0 or step == cfg["steps"]:
            m = acc / (100 if step % 100 == 0 else step % 100)
            rec = {"step": step, "meta": m[0], "zero": m[1], "orth": m[2], "code_norm": m[3],
                   "seconds": time.time() - t0}
            hist.append(rec)
            log(f"[meta] step {step} meta={m[0]:.5f} zero={m[1]:.5f} orth={m[2]:.4f} |z|={m[3]:.3f} "
                f"{time.time() - t0:.0f}s")
            acc[:] = 0
    for p in params:
        p.requires_grad_(False)
    return hist


def selection_blocks(R, lib, cache, data, task, d):
    return make_blocks(R, lib, cache, data.rows_for_task(task, g39_data.SPLIT_VAL), d,
                       seed=SEED + run_loto.SELECTION_OFFSET, draws=NOISE_DRAWS)


def msal(R, lib, cache, data, task, d0, k, weights):
    rows = held_episode_rows(data.episode_indices, data, task, k)
    demo = make_blocks(R, lib, cache, rows, d0, seed=SEED + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
    d_inv = invert_descriptor(R, demo, d0, weights)
    demo_inv = make_blocks(R, lib, cache, rows, d_inv, seed=SEED + FEWSHOT_OFFSET, draws=ORACLE_DRAWS)
    return d_inv, weighted_ls_code(demo_inv, weights)


def offline_scores(R, lib, cache, data, task, d0, weights, shots=SHOTS):
    zero = torch.zeros(K, dtype=torch.float64, device=d0.device)
    out = {"zero": mean_mse(selection_blocks(R, lib, cache, data, task, d0), zero, weights)}
    adapt = {}
    for k in shots:
        d_inv, z = msal(R, lib, cache, data, task, d0, k, weights)
        blocks = selection_blocks(R, lib, cache, data, task, d_inv)
        out[f"msal_k{k}"] = mean_mse(blocks, z, weights)
        out[f"inv_k{k}"] = mean_mse(blocks, zero, weights)
        adapt[k] = (d_inv, z)
    return out, adapt


def setup(h, device):
    data, cache = load(device)
    fit_ids = [t for t in sorted(INSTRUCTIONS) if t != h]
    art = torch.load(RUNS / f"held{h}.pt", weights_only=False, map_location="cpu")
    R, lib = rebuild(art, data, fit_ids, device)
    for p in list(R.parameters()) + list(lib.atom_out.parameters()):
        p.requires_grad_(False)
    R.eval(); lib.eval()
    pr = art["probe"]
    inputs = {k: pr[k] for k in ("state", "x", "t")}
    err0 = float((velocity(R, lib, art["d_star"], None, inputs, device) - pr["zero_shot_v"]).abs().max())
    errs = {k: float((velocity(R, lib, a["d"], a["z"], inputs, device) - a["probe_v"]).abs().max())
            for k, a in art["adapt"].items()}
    if max([err0, *errs.values()]) > 1e-3:
        raise RuntimeError(f"rebuilt R_h disagrees with stored probe: {err0} {errs}")
    return data, cache, fit_ids, art, R, lib, {"zero": err0, **{f"msal_k{k}": v for k, v in errs.items()}}


def make_cfg(args):
    cfg = dict(CFG)
    for key in ("lr", "steps", "train", "query_split"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    return cfg


def mode_probe(args, log):
    data, cache, fit_ids, art, R, lib, errs = setup(args.held, args.device)
    log(f"rebuild probe max-abs errors: {errs}")
    stats = {}
    for t in fit_ids:
        r = data.rows_for_task(t, g39_data.SPLIT_TRAIN)
        _, counts = np.unique(data.episode_indices[r], return_counts=True)
        stats[t] = (len(counts), int(counts.min()), int(np.median(counts)), int(counts.max()))
    log(f"train demos/task min {min(v[0] for v in stats.values())} max {max(v[0] for v in stats.values())}; "
        f"rows/demo median of medians {np.median([v[2] for v in stats.values()])}; "
        f"min {min(v[1] for v in stats.values())} max {max(v[3] for v in stats.values())}")
    for inner in (50, 200):
        cfg = dict(CFG, steps=10, inner_steps=inner)
        torch.cuda.synchronize(); t0 = time.time()
        meta_train(R, lib, data, cache, fit_ids, cfg, args.device, 99, log)
        torch.cuda.synchronize()
        log(f"timing inner_steps={inner}: {(time.time() - t0) / 10:.2f} s/step; "
            f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


def mode_check(args, log):
    data, cache, fit_ids, art, R, lib, errs = setup(args.held, args.device)
    weights = channel_weights(R.config.action_dim, dead_dims=e2.DEAD_ACTION_DIMS, device=args.device)
    rng = np.random.default_rng([SEED, args.held, 5])
    probe_tasks = sorted(int(t) for t in rng.choice(fit_ids, size=2, replace=False))
    train_tasks = [t for t in fit_ids if t not in probe_tasks]
    cfg = make_cfg(args)

    def evaluate(tag):
        res = {}
        for t in probe_tasks:
            dw = torch.tensor(word_descriptor([t]), dtype=torch.float32, device=args.device)
            db = torch.zeros_like(dw); db[0, -1] = 1.0
            res[t] = {"word": offline_scores(R, lib, cache, data, t, dw, weights)[0],
                      "bias_only": offline_scores(R, lib, cache, data, t, db, weights)[0]}
            log(f"[check {tag}] task {t}: {json.dumps(res[t])}")
        return res

    before = evaluate("before")
    hist = meta_train(R, lib, data, cache, train_tasks, cfg, args.device, 1, log)
    after = evaluate("after")
    out = {"held_fold": args.held, "probe_tasks": probe_tasks, "cfg": cfg, "before": before, "after": after,
           "hist": hist}
    (args.out / f"check_fold{args.held}_{cfg['train']}_{cfg['query_split']}_lr{cfg['lr']}_s{cfg['steps']}.json").write_text(json.dumps(out, indent=1))
    log("check done")


def mode_held(args, log):
    t_start = time.time()
    data, cache, fit_ids, art, R, lib, errs = setup(args.held, args.device)
    h = args.held
    weights = channel_weights(R.config.action_dim, dead_dims=e2.DEAD_ACTION_DIMS, device=args.device)
    d_star = torch.tensor(word_descriptor([h]), dtype=torch.float32, device=args.device)
    assert torch.equal(d_star.cpu(), art["d_star"].cpu())
    orig, _ = offline_scores(R, lib, cache, data, h, d_star, weights)
    anchors = {f"msal_k{k}_reproduced": abs(orig[f"msal_k{k}"] - art["adapt"][k]["selection"]) for k in SHOTS}
    log(f"original R_h offline selection: {json.dumps(orig)}; |diff to stored| {anchors}")
    cfg = make_cfg(args)
    t0 = time.time()
    hist = meta_train(R, lib, data, cache, fit_ids, cfg, args.device, 0, log)
    t_train = time.time() - t0
    t1 = time.time()
    new = {"zero": mean_mse(selection_blocks(R, lib, cache, data, h, d_star),
                            torch.zeros(K, dtype=torch.float64, device=args.device), weights)}
    inputs = probe(R, lib, data, data.rows_for_task(h, g39_data.SPLIT_TRAIN), args.device)
    out = {"kind": "held", "held": h, "seed": SEED, "instruction": INSTRUCTIONS[h], "head": head_state(R),
           "library": {k: v.detach().cpu().clone() for k, v in lib.atom_out.state_dict().items()},
           "d_star": d_star.cpu(), "adapt": {}, "finetune": {},
           "anchors": {"rebuild_probe_max_abs": errs, **anchors},
           "probe": {**inputs, "zero_shot_v": velocity(R, lib, d_star, None, inputs, args.device)},
           "meta": {"cfg": cfg, "hist": hist, "source": f"held{h}.pt", "train_seconds": t_train}}
    adapt_seconds = {}
    for k in SHOTS:
        ta = time.time()
        d_inv, z = msal(R, lib, cache, data, h, d_star, k, weights)
        adapt_seconds[k] = time.time() - ta
        blocks = selection_blocks(R, lib, cache, data, h, d_inv)
        new[f"msal_k{k}"] = mean_mse(blocks, z, weights)
        new[f"inv_k{k}"] = mean_mse(blocks, torch.zeros(K, dtype=torch.float64, device=args.device), weights)
        out["adapt"][k] = {"d": d_inv.cpu(), "z": z.cpu(), "selection": new[f"msal_k{k}"],
                           "probe_v": velocity(R, lib, d_inv, z, inputs, args.device)}
    out["meta"].update(offline_original=orig, offline_meta=new, adapt_seconds=adapt_seconds,
                       total_seconds=time.time() - t_start)
    torch.save(out, args.out / f"meta_held{h}.pt")
    rec = {"held": h, "original": orig, "meta": new, "anchors": out["anchors"], "train_seconds": t_train,
           "adapt_seconds": adapt_seconds, "hist": hist, "cfg": cfg}
    with (args.out / "offline.jsonl").open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    log(f"meta R_h offline selection: {json.dumps(new)}")
    log(f"held {h} saved ({time.time() - t_start:.0f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("probe", "check", "held"))
    ap.add_argument("--held", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--train", choices=("all", "film_atoms", "atoms"), default=None)
    ap.add_argument("--query-split", dest="query_split", choices=("train", "val"), default=None)
    args = ap.parse_args()
    if e2.TRAIN_STEPS != 12_800:
        raise RuntimeError("unset G12_TRAIN_STEPS")
    args.out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    {"probe": mode_probe, "check": mode_check, "held": mode_held}[args.mode](args, log)


if __name__ == "__main__":
    main()
