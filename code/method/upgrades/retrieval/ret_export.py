from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from msal_ext.hyb import held_episode_rows
from lib import g39_data

HELD = (1019, 1025, 1000, 1020, 1023, 1030, 1033, 1037)
SHOTS = (1, 16)
RUNS = Path("/home/ubuntu/work/manuscript6/runs/cl_adapt")
N_QUERY = 200


def pad_last_valid(actions: np.ndarray, masks: np.ndarray) -> np.ndarray:
    out = actions.copy()
    for i in range(len(out)):
        valid = np.flatnonzero(masks[i] > 0.5)
        last = valid[-1] if len(valid) else 0
        out[i, last + 1:] = out[i, last]
    return out


def retrieve(q: torch.Tensor, b: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "l2":
        d = (q * q).sum(1, keepdim=True) - 2.0 * q @ b.T + (b * b).sum(1)[None]
        return d.argmin(1)
    qn = q / q.norm(dim=1, keepdim=True).clamp_min(1e-8)
    bn = b / b.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return (qn @ bn.T).argmax(1)


def metric_check(data, mean, std, fold: int, device: str) -> dict:
    res = {}
    for t in [t for t in data.unique_task_ids() if t != fold]:
        val = data.rows_for_task(t, g39_data.SPLIT_VAL)
        q_rows = val[np.linspace(0, len(val) - 1, min(N_QUERY, len(val))).round().astype(int)]
        qf = (torch.from_numpy(data.state_features[q_rows]).to(device).float() - mean) / std
        qa = torch.from_numpy(data.action_chunks[q_rows]).to(device)
        qm = torch.from_numpy(data.action_valid_masks[q_rows]).to(device)
        for k in SHOTS:
            rows = held_episode_rows(data.episode_indices, data, t, k)
            bf = (torch.from_numpy(data.state_features[rows]).to(device).float() - mean) / std
            ba = torch.from_numpy(pad_last_valid(data.action_chunks[rows], data.action_valid_masks[rows])).to(device)
            for metric in ("l2", "cos"):
                idx = retrieve(qf, bf, metric)
                err = ((ba[idx] - qa) ** 2).mean(-1)
                mse = float((err * qm).sum() / qm.sum())
                res.setdefault(f"k{k}_{metric}", {})[t] = mse
            del bf, ba
    return {key: {"mean": float(np.mean(list(v.values()))), "per_task": v} for key, v in res.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = g39_data.load_g39_data(verify_hash=True)
    print(f"[ret_export] data loaded ({time.time() - t0:.0f}s) rows={len(data.task_ids)}", flush=True)
    summary = {"sha256": data.sha256, "banks": {}, "metric_check": {}}
    for h in HELD:
        for k in SHOTS:
            rows = held_episode_rows(data.episode_indices, data, h, k)
            assert (data.task_ids[rows] == h).all() and (data.split_codes[rows] == g39_data.SPLIT_TRAIN).all()
            eps = np.unique(data.episode_indices[rows])
            assert len(eps) == k, (h, k, eps)
            order = np.lexsort((data.step_indices[rows], data.episode_indices[rows]))
            rows = rows[order]
            path = args.out / f"bank_h{h}_k{k}.npz"
            np.savez(path, features=data.state_features[rows].astype(np.float16),
                     actions=data.action_chunks[rows].astype(np.float32),
                     masks=data.action_valid_masks[rows].astype(np.float32),
                     episodes=data.episode_indices[rows], steps=data.step_indices[rows], rows=rows,
                     task=np.int64(h), k=np.int64(k))
            summary["banks"][f"h{h}_k{k}"] = {"rows": int(len(rows)), "episodes": eps.tolist(),
                                              "bytes": int(path.stat().st_size)}
            print(f"[ret_export] bank h={h} k={k}: {len(rows)} rows, episodes {eps.tolist()}", flush=True)
    for h in HELD:
        sd = torch.load(RUNS / f"held{h}.pt", weights_only=False, map_location="cpu")["head"]
        mean, std = sd["state_mean"].float().to(args.device), sd["state_std"].float().to(args.device)
        summary["std_stats"] = summary.get("std_stats", {})
        summary["std_stats"][h] = {"min": float(std.min()), "median": float(std.median()), "max": float(std.max())}
        with torch.no_grad():
            summary["metric_check"][h] = metric_check(data, mean, std, h, args.device)
        mc = summary["metric_check"][h]
        print(f"[ret_export] metric check fold {h}: " + " ".join(f"{key}={v['mean']:.4f}" for key, v in mc.items()),
              flush=True)
    folds = summary["metric_check"]
    rel = []
    for k in SHOTS:
        l2 = np.mean([folds[h][f"k{k}_l2"]["mean"] for h in HELD])
        cos = np.mean([folds[h][f"k{k}_cos"]["mean"] for h in HELD])
        rel.append((l2 - cos) / l2)
        summary[f"k{k}_mean_l2"], summary[f"k{k}_mean_cos"] = float(l2), float(cos)
    summary["cos_relative_gain"] = float(np.mean(rel))
    summary["metric_chosen"] = "cos" if np.mean(rel) > 0.05 else "l2"
    summary["seconds"] = time.time() - t0
    (args.out / "export_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[ret_export] metric chosen: {summary['metric_chosen']} (cos relative gain {np.mean(rel):+.3f}) "
          f"done ({summary['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
