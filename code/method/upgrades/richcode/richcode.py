from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from msal_ext import l40
from msal_ext.hyb import DRAW_STRIDE, FEWSHOT_OFFSET, INVERSION_LR, INVERSION_STEPS, e2, held_episode_rows, \
    run_loto, train_library
from msal_ext.cl_heads import SEED, HostFeatureCache, head_state
from msal_ext.l40 import INSTRUCTIONS, word_descriptor
from joint_sal_velocity import sinusoidal_time_embedding
from lib import g39_data
from lib.structured_field import channel_weights
from run_e35_1_zero_shot_code_loto import NOISE_DRAWS, ORACLE_DRAWS, weighted_ls_code, weighted_mse

HELD = (1019, 1025, 1000, 1020, 1023, 1030, 1033, 1037)
PAIRS = ((1001, 1021), (1002, 1031), (1010, 1022), (1011, 1032))
RUNS = Path("/home/ubuntu/work/manuscript6/runs/cl_adapt")
SLOTS = {"gamma": 512, "beta": 512, "s_state": 512, "s_action": 256, "s_t0": 512, "s_t2": 512}
VARIANTS = {"mod": ("gamma", "beta"), "modsh": tuple(SLOTS)}
GRID_LR = (1e-3, 1e-2)
GRID_LAM = (0.0, 1e-4, 1e-3, 1e-2)
GRID_STEPS = (25, 50, 100, 200, 400)
CHUNK, ACTION = 16, 7


class FHead:

    def __init__(self, sd: dict, lib: dict | None, device: str) -> None:
        self.p = {k: v.to(device).float() for k, v in sd.items()}
        self.L = None if lib is None else {k: v.to(device).float() for k, v in lib.items()}

    def state_pre(self, s):
        p = self.p
        return F.linear((s.float() - p["state_mean"]) / p["state_std"], p["state_encoder.0.weight"],
                        p["state_encoder.0.bias"])

    def hidden(self, spre, x, t, c=None):
        p, c, n = self.p, c or {}, spre.shape[0]
        sp = spre + c["s_state"] if "s_state" in c else spre
        sh = F.layer_norm(F.silu(sp), (512,), p["state_encoder.2.weight"], p["state_encoder.2.bias"])
        a = F.linear(x.reshape(n, -1).float(), p["action_encoder.0.weight"], p["action_encoder.0.bias"])
        ah = F.silu(a + c["s_action"] if "s_action" in c else a)
        h = torch.cat([sh, ah, sinusoidal_time_embedding(t.float(), 64)], dim=-1)
        h = F.linear(h, p["trunk.0.weight"], p["trunk.0.bias"])
        h = F.silu(h + c["s_t0"] if "s_t0" in c else h)
        h = F.linear(h, p["trunk.2.weight"], p["trunk.2.bias"])
        h = F.silu(h + c["s_t2"] if "s_t2" in c else h)
        return F.layer_norm(h, (512,), p["trunk.4.weight"], p["trunk.4.bias"])

    def base(self, h, d, c=None):
        p, c, n = self.p, c or {}, h.shape[0]
        dd = d.reshape(1, -1).expand(n, -1)
        gam, bet = F.linear(dd, p["film_gamma.weight"]), F.linear(dd, p["film_beta.weight"])
        if "gamma" in c:
            gam = gam + c["gamma"]
        if "beta" in c:
            bet = bet + c["beta"]
        g = h * (1.0 + gam) + bet
        out = F.linear(g, p["film_head.weight"], p["film_head.bias"])
        out = out + F.linear(F.silu(F.linear(g, p["film_mlp.inner.weight"], p["film_mlp.inner.bias"])),
                             p["film_mlp.outer.weight"], p["film_mlp.outer.bias"])
        return out.reshape(n, CHUNK, ACTION)

    def atoms(self, h):
        L, n = self.L, h.shape[0]
        inner = F.silu(torch.einsum("kch,bh->bkc", L["w1"], h) + L["b1"][None])
        return (torch.einsum("koc,bkc->bko", L["w2"], inner) + L["b2"][None]).reshape(n, -1, CHUNK, ACTION)

    def velocity(self, s, x, t, d, c=None, z=None):
        h = self.hidden(self.state_pre(s), x, t, c)
        v = self.base(h, d, c)
        if z is not None:
            v = v + torch.einsum("k,bktd->btd", z.float(), self.atoms(h))
        return v


def blocks(fh: FHead, cache, rows, *, seed: int, draws: int) -> list[dict]:
    out = []
    with torch.no_grad():
        for draw in range(draws):
            b = cache.batch(rows, seed=seed + DRAW_STRIDE * draw)
            spre = torch.cat([fh.state_pre(b["state"][i:i + 2048]) for i in range(0, len(rows), 2048)])
            out.append({"spre": spre, "x_t": b["x_t"], "time": b["time"], "target": b["target"], "masks": b["masks"]})
            del b
    return out


def scored_blocks(fh, cache, data, task, *, split=g39_data.SPLIT_VAL):
    return blocks(fh, cache, data.rows_for_task(task, split), seed=SEED + run_loto.SELECTION_OFFSET, draws=NOISE_DRAWS)


def demo_blocks(fh, cache, data, task, k):
    rows = held_episode_rows(data.episode_indices, data, task, k)
    return blocks(fh, cache, rows, seed=SEED + FEWSHOT_OFFSET, draws=ORACLE_DRAWS), int(len(rows))


def ls_blocks(fh, bl, d, c):
    out = []
    with torch.no_grad():
        for b in bl:
            h = fh.hidden(b["spre"], b["x_t"], b["time"], c)
            out.append({"base": fh.base(h, d, c).double(), "atoms": fh.atoms(h).double(),
                        "target": b["target"].double(), "mask": b["masks"].double()})
    return out


def mse(fh, bl, d, c, z, weights) -> float:
    return float(np.mean([weighted_mse(b, z, weights) for b in ls_blocks(fh, bl, d, c)]))


def invert(fh, bl, d0, weights) -> torch.Tensor:
    with torch.no_grad():
        hs = [fh.hidden(b["spre"], b["x_t"], b["time"]) for b in bl]
    d = d0.clone().requires_grad_(True)
    opt = torch.optim.Adam([d], lr=INVERSION_LR)
    for _ in range(INVERSION_STEPS):
        loss = sum(e2.masked_channel_loss(fh.base(h, d), b["target"], b["masks"], weights) for b, h in zip(bl, hs))
        opt.zero_grad(set_to_none=True)
        (loss / len(bl)).backward()
        opt.step()
    return d.detach()


def fit_code(fh, bl, d, variant, lr, lam, steps, weights, on_checkpoint=None):
    names = VARIANTS[variant]
    dev = bl[0]["spre"].device
    c = {n: torch.zeros(SLOTS[n], device=dev, requires_grad=True) for n in names}
    opt = torch.optim.Adam(list(c.values()), lr=lr)
    fixed_h = None
    if variant == "mod":
        with torch.no_grad():
            fixed_h = [fh.hidden(b["spre"], b["x_t"], b["time"]) for b in bl]
    last = {}
    for step in range(1, max(steps) + 1):
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(bl):
            h = fixed_h[i] if fixed_h is not None else fh.hidden(b["spre"], b["x_t"], b["time"], c)
            loss = e2.masked_channel_loss(fh.base(h, d, c), b["target"], b["masks"], weights) / len(bl)
            loss.backward()
        if lam > 0:
            (lam * sum((v ** 2).sum() for v in c.values())).backward()
        opt.step()
        if step in steps:
            last = {n: v.detach().clone() for n, v in c.items()}
            if on_checkpoint is not None:
                on_checkpoint(step, last)
    return last


def code_norm(c) -> float:
    return float(math.sqrt(sum(float((v ** 2).sum()) for v in c.values()))) if c else 0.0


def load_data(device):
    data = g39_data.load_g39_data(verify_hash=True)
    return data, HostFeatureCache(data, device)


def train_proxy(pair, data, cache, device, path: Path):
    fit_ids = [t for t in sorted(INSTRUCTIONS) if t not in HELD and t not in pair]
    assert len(fit_ids) == 30
    t0 = time.time()
    R, _ = run_loto.train_arm("l40_film_nl", seed=SEED, fit_ids=fit_ids, data=data, cache=cache, device=device)
    R.eval()
    lib, loss2 = train_library(R, seed=SEED, fit_ids=fit_ids, data=data, cache=cache, device=device)
    art = {"pair": list(pair), "fit_ids": fit_ids, "seed": SEED, "head": head_state(R),
           "library": {k: v.detach().cpu().clone() for k, v in lib.atom_out.state_dict().items()},
           "stage2_loss": loss2, "seconds": time.time() - t0, "train_steps": e2.TRAIN_STEPS}
    torch.save(art, path)
    print(f"[richcode] proxy head {pair} trained ({art['seconds']:.0f}s)", flush=True)
    return art


def grid_task(fh, data, cache, task, weights, device) -> dict:
    d_star = torch.tensor(word_descriptor([task]), dtype=torch.float32, device=device)
    val = scored_blocks(fh, cache, data, task)
    zero16 = torch.zeros(16, dtype=torch.float64, device=device)
    rec = {"task": task, "zero_shot_val": mse(fh, val, d_star, None, zero16, weights), "k": {}}
    for k in (1, 16):
        demo, nrows = demo_blocks(fh, cache, data, task, k)
        d = invert(fh, demo, d_star, weights)
        z0 = weighted_ls_code(ls_blocks(fh, demo, d, None), weights)
        r = {"demo_rows": nrows, "msal_val": mse(fh, val, d, None, z0, weights),
             "msal_demo": mse(fh, demo, d, None, z0, weights), "configs": {}}
        for variant, lr, lam in itertools.product(VARIANTS, GRID_LR, GRID_LAM):
            res = {}

            def on_ck(step, c):
                z = weighted_ls_code(ls_blocks(fh, demo, d, c), weights)
                res[step] = {"val": mse(fh, val, d, c, z, weights), "demo": mse(fh, demo, d, c, z, weights),
                             "code_norm": code_norm(c)}

            t0 = time.time()
            fit_code(fh, demo, d, variant, lr, lam, GRID_STEPS, weights, on_ck)
            r["configs"][f"{variant}|{lr:g}|{lam:g}"] = {"checkpoints": res, "seconds": time.time() - t0}
        rec["k"][str(k)] = r
        best = min(((n, s, v["val"]) for n, cfg in r["configs"].items() for s, v in cfg["checkpoints"].items()),
                   key=lambda x: x[2])
        print(f"[richcode] task {task} k={k} msal_val={r['msal_val']:.5f} best={best}", flush=True)
        del demo
        torch.cuda.empty_cache()
    return rec


def proxy(args):
    pair = tuple(int(v) for v in args.pair.split(","))
    args.out.mkdir(parents=True, exist_ok=True)
    data, cache = load_data(args.device)
    path = args.out / f"proxyhead_{pair[0]}_{pair[1]}.pt"
    art = torch.load(path, weights_only=False) if path.exists() else train_proxy(pair, data, cache, args.device, path)
    fh = FHead(art["head"], art["library"], args.device)
    weights = channel_weights(7, dead_dims=e2.DEAD_ACTION_DIMS, device=args.device)
    for task in pair:
        out = args.out / f"grid_{task}.json"
        if out.exists():
            continue
        rec = grid_task(fh, data, cache, task, weights, args.device)
        rec["proxy_head"] = path.name
        out.write_text(json.dumps(rec, indent=1))
    print(f"[richcode] proxy {pair} done", flush=True)


def select(args):
    recs = [json.loads(p.read_text()) for p in sorted(args.out.glob("grid_*.json"))]
    out = {"proxy_tasks": [r["task"] for r in recs], "rule": "argmin mean_t log(val_t / msal_val_t)", "k": {}}
    for k in ("1", "16"):
        scores = {}
        for r in recs:
            rk = r["k"][k]
            for name, cfg in rk["configs"].items():
                for step, v in cfg["checkpoints"].items():
                    scores.setdefault((name, int(step)), []).append(math.log(v["val"] / rk["msal_val"]))
        table = {f"{n}|{s}": float(np.mean(v)) for (n, s), v in scores.items() if len(v) == len(recs)}
        best = min(table, key=table.get)
        variant, lr, lam, steps = best.split("|")
        per_variant = {var: min((v, n) for n, v in table.items() if n.startswith(var + "|")) for var in VARIANTS}
        out["k"][k] = {"variant": variant, "lr": float(lr), "lam": float(lam), "steps": int(steps),
                       "mean_log_ratio": table[best], "best_per_variant": per_variant,
                       "msal_over_zero_proxy": float(np.mean([math.log(r["k"][k]["msal_val"] / r["zero_shot_val"])
                                                              for r in recs])),
                       "table": dict(sorted(table.items(), key=lambda x: x[1]))}
    (args.out / "settings.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "table"} for k, v in out["k"].items()}, indent=1))


def held(args):
    settings = json.loads(args.settings.read_text())["k"]
    args.out.mkdir(parents=True, exist_ok=True)
    data, cache = load_data(args.device)
    weights = channel_weights(7, dead_dims=e2.DEAD_ACTION_DIMS, device=args.device)
    tasks = [int(t) for t in args.tasks.split(",")] if args.tasks else list(HELD)
    for h in tasks:
        path = args.out / f"rich{h}.pt"
        if path.exists():
            continue
        t_all = time.time()
        art = torch.load(RUNS / f"held{h}.pt", weights_only=False, map_location="cpu")
        fh = FHead(art["head"], art["library"], args.device)
        pr = art["probe"]
        d_star = art["d_star"].to(args.device)
        val = scored_blocks(fh, cache, data, h)
        out = {"held": h, "seed": SEED, "settings": settings, "probe": {k: pr[k] for k in ("state", "x", "t")},
               "adapt": {}, "anchors": {}}
        with torch.no_grad():
            v0 = fh.velocity(pr["state"].to(args.device), pr["x"].to(args.device), pr["t"].to(args.device), d_star)
        out["anchors"]["zero_probe_max_abs"] = float((v0.cpu() - pr["zero_shot_v"]).abs().max())
        for k in (1, 16):
            s = settings[str(k)]
            ad = art["adapt"][k]
            demo, nrows = demo_blocks(fh, cache, data, h, k)
            t0 = time.time()
            d_re = invert(fh, demo, d_star, weights)
            t_inv = time.time() - t0
            d = ad["d"].to(args.device)
            z0 = weighted_ls_code(ls_blocks(fh, demo, d, None), weights)
            msal_sel = mse(fh, val, d, None, z0, weights)
            with torch.no_grad():
                vm = fh.velocity(pr["state"].to(args.device), pr["x"].to(args.device), pr["t"].to(args.device), d,
                                 None, z0)
            t0 = time.time()
            c = fit_code(fh, demo, d, s["variant"], s["lr"], s["lam"], (s["steps"],), weights)
            z = weighted_ls_code(ls_blocks(fh, demo, d, c), weights)
            t_fit = time.time() - t0
            sel = mse(fh, val, d, c, z, weights)
            with torch.no_grad():
                pv = fh.velocity(pr["state"].to(args.device), pr["x"].to(args.device), pr["t"].to(args.device), d, c, z)
            numbers = 65 + 16 + sum(SLOTS[n] for n in VARIANTS[s["variant"]])
            out["anchors"][f"k{k}"] = {"inversion_reproduced_max_abs": float((d_re - d).abs().max()),
                                       "z_vs_stored_max_abs": float((z0.cpu() - ad["z"].cpu()).abs().max()),
                                       "msal_selection_recomputed": msal_sel, "msal_selection_stored": ad["selection"],
                                       "msal_probe_max_abs": float((vm.cpu() - ad["probe_v"]).abs().max())}
            out["adapt"][k] = {"variant": s["variant"], "d": d.cpu(), "z": z.cpu(),
                               "code": {n: v.cpu() for n, v in c.items()}, "selection": sel,
                               "msal_selection": ad["selection"], "ft_selection": art["finetune"][k]["selection"],
                               "demo_mse": mse(fh, demo, d, c, z, weights), "msal_demo_mse": mse(fh, demo, d, None, z0, weights),
                               "code_norm": code_norm(c), "z_norm": float(z.norm()), "demo_rows": nrows,
                               "numbers_per_task": numbers, "seconds_inversion": t_inv, "seconds_code": t_fit,
                               "probe_v": pv.cpu()}
            print(f"[richcode] held {h} k={k} {s['variant']} sel={sel:.5f} msal={ad['selection']:.5f} "
                  f"(recomputed {msal_sel:.5f}) ft={art['finetune'][k]['selection']:.5f} "
                  f"demo={out['adapt'][k]['demo_mse']:.5f}/{out['adapt'][k]['msal_demo_mse']:.5f} "
                  f"|c|={code_norm(c):.3f} anchors={out['anchors'][f'k{k}']}", flush=True)
            del demo
        out["seconds"] = time.time() - t_all
        torch.save(out, path)
        del art, fh, val
        torch.cuda.empty_cache()
    print("[richcode] held done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("proxy", "select", "held"))
    ap.add_argument("--pair", default="")
    ap.add_argument("--settings", type=Path)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke-steps", type=int, default=0)
    args = ap.parse_args()
    if e2.TRAIN_STEPS != 12_800:
        raise RuntimeError("unset G12_TRAIN_STEPS")
    if args.smoke_steps:
        global GRID_LR, GRID_LAM, GRID_STEPS
        e2.TRAIN_STEPS = int(args.smoke_steps)
        GRID_LR, GRID_LAM, GRID_STEPS = (1e-2,), (0.0, 1e-3), (25, 50)
    {"proxy": proxy, "select": select, "held": held}[args.mode](args)


if __name__ == "__main__":
    main()
