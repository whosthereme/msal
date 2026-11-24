from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

W = Path("/home/ubuntu/work/manuscript6")
SHARDS = W / "repo/sal_v2_identifiable/runs/20260829_g39_libero40_substrate_v1/shards"
RUNS = W / "runs/cl_adapt"
CHUNK, ACTION, STEPS = 16, 7, 10
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2
HELD = (1019, 1025, 1000, 1020, 1023, 1030, 1033, 1037)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(task: int, *, split: int, first_episodes: int | None = None) -> dict:
    z = np.load(SHARDS / f"task{task}.npz")
    sp, ep = z["split_codes"], z["episode_indices"]
    assert np.all(z["task_ids"] == task)
    rows = np.flatnonzero(sp == split)
    if first_episodes is not None:
        chosen = np.sort(np.unique(ep[rows]))[:first_episodes]
        rows = rows[np.isin(ep[rows], chosen)]
    return {"state": torch.from_numpy(z["state_features"][rows].astype(np.float32)),
            "actions": torch.from_numpy(z["action_chunks"][rows].astype(np.float32)),
            "masks": torch.from_numpy(z["action_valid_masks"][rows].astype(np.float32)),
            "episodes": ep[rows].copy(), "task": task, "split": split}


def demo_rows(task: int, k: int, rows16: dict | None = None) -> dict:
    r = rows16 if rows16 is not None else load_rows(task, split=SPLIT_TRAIN, first_episodes=k)
    chosen = np.sort(np.unique(r["episodes"]))[:k]
    keep = np.isin(r["episodes"], chosen)
    idx = torch.from_numpy(np.flatnonzero(keep))
    return {**{key: r[key][idx] for key in ("state", "actions", "masks")}, "episodes": r["episodes"][keep],
            "task": task, "split": r["split"]}


def time_embedding(t: torch.Tensor, dimension: int = 64) -> torch.Tensor:
    freqs = torch.exp(torch.linspace(0.0, -math.log(10_000.0), dimension // 2, device=t.device))
    angles = t[:, None] * freqs[None, :] * 2.0 * math.pi
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


class FHead:
    def __init__(self, sd: dict, lib: dict | None, device: str):
        self.p = {k: v.float().to(device) for k, v in sd.items()}
        self.L = None if lib is None else {k: v.float().to(device) for k, v in lib.items()}
        self.device = device

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        p = self.p
        out = []
        with torch.no_grad():
            for i in range(0, s.shape[0], 512):
                x = (s[i:i + 512].to(self.device).float() - p["state_mean"]) / p["state_std"]
                out.append(F.layer_norm(F.silu(F.linear(x, p["state_encoder.0.weight"], p["state_encoder.0.bias"])),
                                        (512,), p["state_encoder.2.weight"], p["state_encoder.2.bias"]))
        return torch.cat(out)

    def hidden(self, sh, x, t):
        p = self.p
        n = sh.shape[0]
        ah = F.silu(F.linear(x.reshape(n, -1), p["action_encoder.0.weight"], p["action_encoder.0.bias"]))
        h = torch.cat([sh, ah, time_embedding(t)], dim=-1)
        h = F.silu(F.linear(h, p["trunk.0.weight"], p["trunk.0.bias"]))
        h = F.silu(F.linear(h, p["trunk.2.weight"], p["trunk.2.bias"]))
        return F.layer_norm(h, (512,), p["trunk.4.weight"], p["trunk.4.bias"])

    def base_from_hidden(self, h, d):
        p = self.p
        n = h.shape[0]
        dd = d.reshape(1, -1).float().expand(n, -1)
        g = h * (1.0 + F.linear(dd, p["film_gamma.weight"])) + F.linear(dd, p["film_beta.weight"])
        out = F.linear(g, p["film_head.weight"], p["film_head.bias"])
        out = out + F.linear(F.silu(F.linear(g, p["film_mlp.inner.weight"], p["film_mlp.inner.bias"])),
                             p["film_mlp.outer.weight"], p["film_mlp.outer.bias"])
        return out.reshape(n, CHUNK, ACTION)

    def atoms(self, h):
        L = self.L
        n = h.shape[0]
        inner = F.silu(torch.einsum("kch,bh->bkc", L["w1"], h) + L["b1"][None])
        return (torch.einsum("koc,bkc->bko", L["w2"], inner) + L["b2"][None]).reshape(n, -1, CHUNK, ACTION)

    def velocity(self, sh, x, t, d, z=None):
        h = self.hidden(sh, x, t)
        v = self.base_from_hidden(h, d)
        if z is not None:
            v = v + torch.einsum("k,bktd->btd", z.float(), self.atoms(h))
        return v

    def velocity_raw(self, s, x, t, d, z=None):
        return self.velocity(self.encode(s), x.to(self.device).float(), t.to(self.device).float(), d.to(self.device), z)

    def sample(self, sh, eps, d, z=None, steps: int = STEPS):
        x = eps
        n = sh.shape[0]
        for i in range(steps):
            t = torch.full((n,), 1.0 - i / steps, device=sh.device)
            x = x - self.velocity(sh, x, t, d, z) / steps
        return x


def masked_mse(pred, target, mask):
    per_row = ((pred - target) ** 2).mean(dim=-1)
    return (per_row * mask).sum() / mask.sum().clamp_min(1.0)


def noise(n: int, draws: int, seed: int, device: str) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(draws, n, CHUNK, ACTION, generator=g).to(device)


def action_error(head, sh, actions, masks, d, z, eps, clip: bool = True, batch: int = 4096) -> float:
    vals = []
    with torch.no_grad():
        for j in range(eps.shape[0]):
            x = torch.cat([head.sample(sh[i:i + batch], eps[j, i:i + batch], d, z) for i in range(0, sh.shape[0], batch)])
            if clip:
                x = x.clamp(-1.0, 1.0)
            vals.append(float(masked_mse(x, actions, masks)))
    return float(np.mean(vals))


def flow_blocks(n: int, draws: int, seed: int, device: str, tau_range=(0.001, 0.999)):
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    out = []
    for _ in range(draws):
        e = torch.randn(n, CHUNK, ACTION, generator=g)
        t = torch.rand(n, generator=g) * (tau_range[1] - tau_range[0]) + tau_range[0]
        out.append((e.to(device), t.to(device)))
    return out


def flow_mse(head, sh, actions, masks, d, z, blocks) -> float:
    vals = []
    with torch.no_grad():
        for e, t in blocks:
            x_t = t[:, None, None] * e + (1 - t[:, None, None]) * actions
            vals.append(float(masked_mse(head.velocity(sh, x_t, t, d, z), e - actions, masks)))
    return float(np.mean(vals))


def invert_and_code(head, sh, actions, masks, d0, blocks, steps=200, lr=1e-2, with_code=True):
    with torch.no_grad():
        hs = []
        for e, t in blocks:
            x_t = t[:, None, None] * e + (1 - t[:, None, None]) * actions
            hs.append((head.hidden(sh, x_t, t), e - actions))
    d = d0.clone().float().reshape(1, -1).requires_grad_(True)
    opt = torch.optim.Adam([d], lr=lr)
    for _ in range(steps):
        loss = sum(masked_mse(head.base_from_hidden(h, d), tgt, masks) for h, tgt in hs) / len(hs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    d = d.detach()
    if not with_code:
        return d, None
    K = head.L["w1"].shape[0]
    normal = torch.zeros(K, K, dtype=torch.float64, device=sh.device)
    rhs = torch.zeros(K, dtype=torch.float64, device=sh.device)
    with torch.no_grad():
        for h, tgt in hs:
            base = head.base_from_hidden(h, d).double()
            at = head.atoms(h).double()
            scale = masks.double().sqrt()[:, :, None]
            design = (at * scale[:, None]).permute(0, 2, 3, 1).reshape(-1, K)
            resid = ((tgt.double() - base) * scale).reshape(-1)
            bs = 1.0 / float(masks.sum().clamp_min(1.0))
            normal += bs * design.T @ design
            rhs += bs * design.T @ resid
    z = torch.linalg.lstsq(normal, rhs[:, None]).solution[:, 0]
    return d, z


def action_fit(head, sh, actions, masks, d_init, z_init, eps, *, steps, lr_d, lr_z, checkpoints=(), anchor=0.0,
               batch: int = 2048, log=None):
    d = d_init.clone().float().reshape(1, -1).requires_grad_(True)
    z = z_init.clone().float().requires_grad_(True)
    d0, z0 = d_init.clone().float().reshape(1, -1), z_init.clone().float()
    opt = torch.optim.Adam([{"params": [d], "lr": lr_d}, {"params": [z], "lr": lr_z}])
    n = sh.shape[0]
    pairs = [(j, i) for j in range(eps.shape[0]) for i in range(0, n, batch)]
    total = float(masks.sum()) * eps.shape[0]
    out = {}
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for j, i in pairs:
            x = head.sample(sh[i:i + batch], eps[j, i:i + batch], d, z)
            part = (((x - actions[i:i + batch]) ** 2).mean(-1) * masks[i:i + batch]).sum() / total
            part.backward()
            acc += float(part.detach())
        if anchor > 0:
            reg = anchor * (((d - d0) ** 2).sum() + ((z - z0) ** 2).sum())
            reg.backward()
        opt.step()
        if log is not None and (step == 1 or step % 50 == 0):
            log(f"    step {step} demo_action_loss {acc:.5f}")
        if step in checkpoints or step == steps:
            out[step] = (d.detach().clone(), z.detach().clone().double())
    return out


def action_fit_lm(head, sh, actions, masks, d_init, z_init, eps, *, iters, lam0=1e-2, checkpoints=(),
                  batch: int = 2048, log=None):
    from torch.func import jvp
    nd = d_init.numel()
    p = torch.cat([d_init.reshape(-1).float(), z_init.reshape(-1).float()]).to(sh.device)
    P = p.numel()
    n = sh.shape[0]
    total = float(masks.sum()) * eps.shape[0] * ACTION
    w = (masks / total).sqrt()[:, :, None]

    def resid(p, j, i):
        x = head.sample(sh[i:i + batch], eps[j, i:i + batch], p[:nd].reshape(1, -1), p[nd:])
        return ((x - actions[i:i + batch]) * w[i:i + batch]).reshape(-1)

    parts = [(j, i) for j in range(eps.shape[0]) for i in range(0, n, batch)]

    def loss_of(p):
        with torch.no_grad():
            return float(sum(resid(p, j, i).double().pow(2).sum() for j, i in parts))

    lam, cur, out = lam0, loss_of(p), {}
    eye = torch.eye(P, device=sh.device)
    for it in range(1, iters + 1):
        JtJ = torch.zeros(P, P, dtype=torch.float64, device=sh.device)
        Jtr = torch.zeros(P, dtype=torch.float64, device=sh.device)
        for j, i in parts:
            cols, r = [], None
            for q in range(P):
                r, col = jvp(lambda pp: resid(pp, j, i), (p,), (eye[q],))
                cols.append(col.double())
            J = torch.stack(cols, dim=1)
            JtJ += J.T @ J
            Jtr += J.T @ r.double()
        diag = torch.diag(JtJ).clamp_min(1e-12)
        for _ in range(12):
            step = torch.linalg.solve(JtJ + lam * torch.diag(diag), -Jtr)
            cand = p + step.float()
            new = loss_of(cand)
            if new < cur:
                p, cur, lam = cand, new, max(lam / 3, 1e-6)
                break
            lam *= 4
        if log is not None:
            log(f"    LM it {it} demo_action_mse {cur:.5f} lam {lam:.2e}")
        if it in checkpoints or it == iters:
            out[it] = (p[:nd].reshape(1, -1).clone(), p[nd:].clone().double())
    return out


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    for arr, r in ((a, ra), (b, rb)):
        for v in np.unique(arr):
            m = arr == v
            if m.sum() > 1:
                r[m] = r[m].mean()
    return float(np.corrcoef(ra, rb)[0, 1])
