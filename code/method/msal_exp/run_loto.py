from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT.parent, ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import run_e2_structured_training as e2
from joint_sal_velocity import output_orthogonality_error
from lib.frozen_data import DeviceCache
from lib.pool_v5 import source_task_ids
from lib.structured_field import channel_weights, group_sparsity_penalty
from run_e35_1_zero_shot_code_loto import (
    NOISE_DRAWS,
    ORACLE_DRAWS,
    ORACLE_ROWS_CAP,
    collect_blocks,
    nearest_source_task,
    weighted_ls_code,
    weighted_mse,
)

from msal_exp.arms import (
    ARMS,
    CODE_AFFINE_FAMILIES,
    DESCRIPTORS,
    build_model,
    config_for,
    parameter_report,
)
from msal_exp.data_lang import (
    LANG_NPZ,
    SPLIT_ROBUSTNESS,
    SPLIT_SELECTION,
    SPLIT_TRAIN,
    load_lang_v5,
)

PROTOCOL = "revision_20261021_loto_v1"
REFERENCE_DIR = ROOT.parent.parent / "reference"
SELECTION_OFFSET = 70_000
ROBUSTNESS_OFFSET = 90_000
DRAW_STRIDE = 1_000_003


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes() -> dict[str, str]:
    files = sorted((ROOT / "msal_exp").glob("*.py")) + sorted((ROOT / "lib").glob("*.py"))
    files += [ROOT / "scripts" / "run_e2_structured_training.py", ROOT / "scripts" / "run_e35_1_zero_shot_code_loto.py"]
    files += [ROOT.parent / "joint_sal_velocity.py"]
    return {str(p.relative_to(ROOT.parent)): file_sha(p) for p in files}


def environment() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "G12_TRAIN_STEPS_env": os.environ.get("G12_TRAIN_STEPS"),
        "train_steps_effective": int(e2.TRAIN_STEPS),
        "batch_size": e2.BATCH_SIZE,
        "learning_rate": e2.LEARNING_RATE,
        "weight_decay": e2.WEIGHT_DECAY,
        "orthogonality_weight": e2.ORTHOGONALITY_WEIGHT,
        "group_sparsity_weight": e2.GROUP_SPARSITY_WEIGHT,
        "dead_action_dims": list(e2.DEAD_ACTION_DIMS),
        "noise_draws": NOISE_DRAWS,
    }


def train_arm(arm: str, *, seed: int, fit_ids: list[int], data, cache, device: str):
    spec = ARMS[arm]
    family = spec["family"]
    lookup = {task_id: index for index, task_id in enumerate(fit_ids)}
    torch.manual_seed(seed)
    config = config_for(arm, data)
    model = build_model(arm, config, fit_ids).to(device)
    weights = channel_weights(config.action_dim, dead_dims=e2.DEAD_ACTION_DIMS, device=device)
    rows_by_task = {t: data.rows_for_task(t, SPLIT_TRAIN) for t in fit_ids}
    for task_id, rows in rows_by_task.items():
        if len(rows) == 0:
            raise RuntimeError(f"task {task_id} has no training rows")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=e2.LEARNING_RATE,
        weight_decay=e2.WEIGHT_DECAY,
    )
    rng = np.random.default_rng(seed)
    per_task = max(e2.BATCH_SIZE // len(fit_ids), 1)
    for step in range(1, e2.TRAIN_STEPS + 1):
        picked = np.concatenate([rng.choice(rows_by_task[t], size=per_task, replace=True) for t in fit_ids])
        batch = cache.batch(picked, seed=seed * 7919 + step)
        task_rows = torch.tensor([lookup[int(v)] for v in data.task_ids[picked]], device=device)
        if family in CODE_AFFINE_FAMILIES:
            codes = model.codes_for_rows(task_rows)
            base, atoms = model.components(batch["state"], batch["x_t"], batch["time"], None)
            prediction = base + torch.einsum("bk,bktd->btd", codes, atoms)
            fit = e2.masked_channel_loss(prediction, batch["target"], batch["masks"], weights)
            if spec["reg"]:
                loss = (
                    fit
                    + e2.ORTHOGONALITY_WEIGHT * output_orthogonality_error(atoms)
                    + e2.GROUP_SPARSITY_WEIGHT * group_sparsity_penalty(model.structured_codes, atoms)
                )
            else:
                loss = fit
        else:
            prediction = model.forward_with_rows(batch["state"], batch["x_t"], batch["time"], task_rows)
            loss = e2.masked_channel_loss(prediction, batch["target"], batch["masks"], weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model, lookup


def score_code_affine(model, *, held: int, fit_ids: list[int], seed: int, data, cache, device: str) -> dict:
    with torch.no_grad():
        phi = model.codes_for_task_ids([held])[0].detach()
        source = model.codes_for_task_ids(fit_ids).detach()
    mean_code = source.mean(dim=0)
    neighbour, neighbour_distance = nearest_source_task(held, fit_ids)
    nn_code = source[fit_ids.index(neighbour)]
    weights = channel_weights(int(data.action_chunks.shape[2]), dead_dims=e2.DEAD_ACTION_DIMS, device=device)

    scored = collect_blocks(model, data, cache, held, split=SPLIT_SELECTION, seed=seed + SELECTION_OFFSET, draws=NOISE_DRAWS)
    train_blocks = collect_blocks(
        model, data, cache, held, split=SPLIT_TRAIN, seed=seed + 60_000, draws=ORACLE_DRAWS, rows_cap=ORACLE_ROWS_CAP
    )
    oracle = weighted_ls_code(train_blocks, weights)
    del train_blocks
    oracle_absolute = weighted_ls_code(scored, weights)
    codes = {"phi": phi, "nn": nn_code, "mean": mean_code, "oracle": oracle, "oracle_absolute": oracle_absolute}
    selection = {name: float(np.mean([weighted_mse(b, c, weights) for b in scored])) for name, c in codes.items()}
    del scored

    robust = collect_blocks(model, data, cache, held, split=SPLIT_ROBUSTNESS, seed=seed + ROBUSTNESS_OFFSET, draws=NOISE_DRAWS)
    robustness = {
        name: float(np.mean([weighted_mse(b, codes[name], weights) for b in robust]))
        for name in ("phi", "nn", "mean", "oracle")
    }
    del robust
    return {
        "selection": selection,
        "robustness": robustness,
        "main_selection": selection["phi"],
        "main_robustness": robustness["phi"],
        "nearest_source_task": int(neighbour),
        "nearest_source_distance_m": float(neighbour_distance),
        "code_norms": {name: float(code.norm()) for name, code in codes.items()},
    }


def _forward_mse(model, arm: str, *, rows: np.ndarray, seed: int, offset: int, cache, device: str, held_row=None, descriptor=None) -> float:
    family = ARMS[arm]["family"]
    weights = channel_weights(model.config.action_dim, dead_dims=e2.DEAD_ACTION_DIMS, device=device)
    total, count = 0.0, 0
    with torch.no_grad():
        for draw in range(NOISE_DRAWS):
            batch = cache.batch(rows, seed=seed + offset + DRAW_STRIDE * draw)
            n = batch["state"].shape[0]
            if family == "context":
                prediction = model.forward_with_rows(batch["state"], batch["x_t"], batch["time"], held_row.expand(n))
            elif family == "film":
                prediction = model.forward_descriptor(batch["state"], batch["x_t"], batch["time"], descriptor.expand(n, -1))
            else:
                prediction = model.forward_descriptor(batch["state"], batch["x_t"], batch["time"])
            error = (prediction - batch["target"]) ** 2 * weights[None, None, :]
            per_row = error.sum(dim=-1) / weights.sum().clamp_min(1e-12)
            mask = batch["masks"]
            total += float((per_row * mask).sum() / mask.sum().clamp_min(1.0))
            count += 1
    return total / max(count, 1)


def score_forward(model, arm: str, *, held: int, seed: int, data, cache, device: str) -> dict:
    family = ARMS[arm]["family"]
    held_row = descriptor = None
    if family == "context":
        from lib.context_cond import build_context_features

        held_features = build_context_features([held], arm="context_phi", feature_set="saturated_v5", pool="v5", radius_count=4)
        with torch.no_grad():
            model.task_features = torch.cat(
                [model.task_features, torch.tensor(held_features, dtype=torch.float32, device=device)], dim=0
            )
        held_row = torch.tensor([model.task_features.shape[0] - 1], device=device, dtype=torch.long)
    elif family == "film":
        matrix = DESCRIPTORS[ARMS[arm]["descriptor"]]([held])
        descriptor = torch.tensor(matrix, dtype=torch.float32, device=device)
    out = {}
    for name, split, offset in (("selection", SPLIT_SELECTION, SELECTION_OFFSET), ("robustness", SPLIT_ROBUSTNESS, ROBUSTNESS_OFFSET)):
        rows = data.rows_for_task(held, split)
        if len(rows) == 0:
            raise RuntimeError(f"held task {held} has no rows in split {split}")
        out[name] = _forward_mse(model, arm, rows=rows, seed=seed, offset=offset, cache=cache, device=device, held_row=held_row, descriptor=descriptor)
    return {"selection": {"main": out["selection"]}, "robustness": {"main": out["robustness"]}, "main_selection": out["selection"], "main_robustness": out["robustness"]}


def load_references(seed: int) -> dict[str, dict[int, Any]]:
    refs: dict[str, dict[int, Any]] = {}
    e35 = REFERENCE_DIR / f"e35_1_seed{seed}" / "gate_report.json"
    if e35.exists():
        refs["sal"] = {int(f["held_task"]): f["mse"] for f in json.loads(e35.read_text())["folds"]}
    e40 = REFERENCE_DIR / f"e40_1_seed{seed}" / "gate_report.json"
    if e40.exists():
        refs["ctx_add"] = {
            int(f["held_task"]): float(f["mse"]) for f in json.loads(e40.read_text())["folds"] if f["arm"] == "context_phi"
        }
    return refs


def run_anchor(*, seed: int, data, cache, device: str) -> dict:
    path = REFERENCE_DIR / f"e33_1_seed{seed}" / "gate_report.json"
    published = float(json.loads(path.read_text())["selection_loss"])
    fit_ids = list(source_task_ids())
    started = time.time()
    model, lookup = train_arm("sal", seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
    weights = channel_weights(int(data.action_chunks.shape[2]), dead_dims=e2.DEAD_ACTION_DIMS, device=device)
    reproduced = e2.validation_loss(model, data, cache, data.rows_for_split(SPLIT_SELECTION), lookup, weights, seed + 20_000, device)
    gap = abs(reproduced - published)
    return {
        "seed": seed,
        "published_selection_loss_e33_1": published,
        "reproduced_selection_loss": reproduced,
        "absolute_gap": gap,
        "relative_gap": gap / max(abs(published), 1e-30),
        "bit_exact": bool(gap == 0.0),
        "wall_clock_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", choices=("blind", "lang"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arms", required=True, help="comma-separated arm names from msal_exp.arms.ARMS")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--held-tasks", default="", help="comma-separated subset (smoke); default all 41")
    parser.add_argument("--anchor", action="store_true", help="also run the all-41 E33.1 selection-loss anchor first")
    parser.add_argument("--skip-hash", action="store_true", help="skip the 3.1 GB data digest (smoke only)")
    parser.add_argument("--smoke-steps", type=int, default=0, help="bug-check only: train for N updates; output is marked smoke")
    args = parser.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; choose from {sorted(ARMS)}")
    if e2.TRAIN_STEPS != 12_800:
        raise RuntimeError(f"TRAIN_STEPS is {e2.TRAIN_STEPS}, not the published 12,800; unset G12_TRAIN_STEPS")
    if args.smoke_steps:
        if args.anchor:
            raise ValueError("--smoke-steps cannot be combined with --anchor")
        e2.TRAIN_STEPS = int(args.smoke_steps)
        args.output = args.output.with_name(args.output.name + "_SMOKE")

    args.output.mkdir(parents=True, exist_ok=True)
    folds_path = args.output / "folds.jsonl"
    done = set()
    if folds_path.exists():
        for line in folds_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                done.add((record["arm"], int(record["held_task"])))

    data = load_lang_v5(args.feature, verify_hash=not args.skip_hash)
    cache = DeviceCache(data, args.device)
    data.state_features = None

    run_info = {
        "protocol": PROTOCOL,
        "feature": args.feature,
        "seed": args.seed,
        "arms": arms,
        "data_path": str(LANG_NPZ),
        "data_sha256": data.sha256 or "not_verified_smoke",
        "environment": environment(),
        "code_sha256": code_hashes(),
        "started_unix": time.time(),
        "rollouts_executed": 0,
        "sealed_rows_read": 0,
        "smoke": bool(args.smoke_steps),
    }
    run_info_path = args.output / "run_info.json"
    if not run_info_path.exists():
        run_info_path.write_text(json.dumps(run_info, indent=1))

    if args.anchor and not (args.output / "anchor.json").exists():
        anchor = run_anchor(seed=args.seed, data=data, cache=cache, device=args.device)
        (args.output / "anchor.json").write_text(json.dumps(anchor, indent=1))
        print(f"anchor seed {args.seed}: published {anchor['published_selection_loss_e33_1']:.10f} "
              f"reproduced {anchor['reproduced_selection_loss']:.10f} gap {anchor['absolute_gap']:.3e}", flush=True)
        torch.cuda.empty_cache()

    references = load_references(args.seed) if args.feature == "blind" else {}
    all_ids = list(source_task_ids())
    held_list = [int(t) for t in args.held_tasks.split(",") if t.strip()] if args.held_tasks else all_ids

    for held in held_list:
        fit_ids = [t for t in all_ids if t != held]
        if len(fit_ids) != len(all_ids) - 1:
            raise RuntimeError(f"held task {held} is not a v5 source task")
        for arm in arms:
            if (arm, held) in done:
                continue
            started = time.time()
            model, _ = train_arm(arm, seed=args.seed, fit_ids=fit_ids, data=data, cache=cache, device=args.device)
            train_seconds = time.time() - started
            model.eval()
            if ARMS[arm]["family"] in CODE_AFFINE_FAMILIES:
                scores = score_code_affine(model, held=held, fit_ids=fit_ids, seed=args.seed, data=data, cache=cache, device=args.device)
            else:
                scores = score_forward(model, arm, held=held, seed=args.seed, data=data, cache=cache, device=args.device)
            params = parameter_report(model, arm)
            del model
            torch.cuda.empty_cache()
            record = {
                "arm": arm,
                "feature": args.feature,
                "seed": args.seed,
                "held_task": held,
                "train_steps": int(e2.TRAIN_STEPS),
                **scores,
                **params,
                "train_seconds": train_seconds,
                "wall_clock_seconds": time.time() - started,
            }
            if arm in references and held in references[arm]:
                record["reference"] = references[arm][held]
            with folds_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            ref = record.get("reference")
            ref_text = ""
            if isinstance(ref, dict):
                ref_text = f" ref_phi={ref['phi']:.10f} diff={record['main_selection'] - ref['phi']:+.2e}"
            elif isinstance(ref, float):
                ref_text = f" ref={ref:.10f} diff={record['main_selection'] - ref:+.2e}"
            print(f"[{args.feature}/{arm}] seed={args.seed} held={held:2d} sel={record['main_selection']:.10f} "
                  f"rob={record['main_robustness']:.6f}{ref_text} {record['wall_clock_seconds']:.0f}s", flush=True)

    records = [json.loads(line) for line in folds_path.read_text().splitlines() if line.strip()]
    summary = {"protocol": PROTOCOL, "feature": args.feature, "seed": args.seed, "arms": {}}
    for arm in arms:
        members = [r for r in records if r["arm"] == arm]
        entry = {
            "fold_count": len(members),
            "complete": len({r["held_task"] for r in members}) == len(held_list),
            "mean_selection": float(np.mean([r["main_selection"] for r in members])) if members else None,
            "mean_robustness": float(np.mean([r["main_robustness"] for r in members])) if members else None,
        }
        refd = [r for r in members if "reference" in r]
        if refd:
            diffs = [r["main_selection"] - (r["reference"]["phi"] if isinstance(r["reference"], dict) else r["reference"]) for r in refd]
            entry["reference_max_abs_diff"] = float(np.max(np.abs(diffs)))
            entry["reference_folds"] = len(refd)
        summary["arms"][arm] = entry
    summary["finished_unix"] = time.time()
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
