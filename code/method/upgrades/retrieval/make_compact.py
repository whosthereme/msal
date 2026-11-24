from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ret_rollout import Head, state_code

HELD = (1019, 1025, 1000, 1020, 1023, 1030, 1033, 1037)


def pad_last_valid(actions: np.ndarray, masks: np.ndarray) -> np.ndarray:
    out = actions.copy()
    for i in range(len(out)):
        valid = np.flatnonzero(masks[i] > 0.5)
        last = valid[-1] if len(valid) else 0
        out[i, last + 1:] = out[i, last]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--banks", type=Path, required=True)
    args = ap.parse_args()
    for h in HELD:
        head = Head(torch.load(args.runs / f"held{h}.pt", weights_only=False, map_location="cpu")["head"])
        b = np.load(args.banks / f"bank_h{h}_k1.npz")
        with torch.no_grad():
            sh = state_code(head, torch.from_numpy(b["features"]).float()).numpy().astype(np.float16)
        acts = pad_last_valid(b["actions"], b["masks"]).astype(np.float32)
        out = args.banks / f"bank_sh_h{h}_k1.npz"
        np.savez(out, sh=sh, actions=acts, episodes=b["episodes"], steps=b["steps"], task=b["task"], k=b["k"])
        full = b["features"].nbytes + b["actions"].nbytes + b["masks"].nbytes
        print(f"h={h} rows={len(sh)} compact={sh.nbytes + acts.nbytes} B full={full} B ratio={full / (sh.nbytes + acts.nbytes):.1f}",
              flush=True)


if __name__ == "__main__":
    main()
