from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from msal_exp import run_loto
from msal_exp import arms
from msal_ext.nl import (
    DescriptorMoE,
    FiLMNonlinearReadout,
    NonlinearSAL,
    PlainNonlinearReadout,
)
from lib.nonlinear_atom_field import NonlinearAtomHead, linear_map_residual

DIAGNOSTIC_ROWS = 1_024
DIAGNOSTIC_OFFSET = 123_457

arms.ARMS.update({
    "sal_nl": {"family": "sal_v5", "descriptor": "saturated", "reg": True},
    "sal_nl_noreg": {"family": "sal_v5", "descriptor": "saturated", "reg": False},
    "film_nl": {"family": "film", "descriptor": "saturated", "reg": False},
    "plain_nl": {"family": "plain", "descriptor": None, "reg": False},
    "moe_desc": {"family": "film", "descriptor": "saturated", "reg": False},
})
BUILDERS = {
    "sal_nl": lambda config, ids: NonlinearSAL(config, ids),
    "sal_nl_noreg": lambda config, ids: NonlinearSAL(config, ids),
    "film_nl": lambda config, ids: FiLMNonlinearReadout(config, ids, arms.DESCRIPTORS["saturated"]),
    "plain_nl": lambda config, ids: PlainNonlinearReadout(config, ids),
    "moe_desc": lambda config, ids: DescriptorMoE(config, ids, arms.DESCRIPTORS["saturated"]),
}
CONDITIONED = {
    "sal_nl": ("atom_out", "code_map"),
    "sal_nl_noreg": ("atom_out", "code_map"),
    "film_nl": ("film_gamma", "film_beta"),
    "plain_nl": (),
    "moe_desc": ("atom_out", "gate"),
}

_build_model = arms.build_model
_parameter_report = arms.parameter_report
_code_hashes = run_loto.code_hashes
_score_code_affine = run_loto.score_code_affine


def build_model(arm, config, fit_ids):
    if arm in BUILDERS:
        return BUILDERS[arm](config, fit_ids)
    return _build_model(arm, config, fit_ids)


def parameter_report(model, arm):
    if arm not in BUILDERS:
        return _parameter_report(model, arm)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    prefixes = CONDITIONED[arm]
    return {
        "trainable_parameters": int(sum(p.numel() for p in trainable.values())),
        "task_conditioned_parameters": int(sum(p.numel() for n, p in trainable.items() if prefixes and n.startswith(prefixes))),
        "nonlinear_readout_parameters": int(sum(p.numel() for n, p in trainable.items() if n.startswith(("film_mlp", "plain_mlp")))),
    }


def score_code_affine(model, *, held, fit_ids, seed, data, cache, device):
    scores = _score_code_affine(model, held=held, fit_ids=fit_ids, seed=seed, data=data, cache=cache, device=device)
    if isinstance(model.atom_out, NonlinearAtomHead):
        rng = np.random.default_rng(seed + DIAGNOSTIC_OFFSET)
        pool = np.concatenate([data.rows_for_task(t, run_loto.SPLIT_TRAIN) for t in fit_ids])
        rows = np.sort(rng.choice(pool, size=DIAGNOSTIC_ROWS, replace=False))
        batch = cache.batch(rows, seed=seed + DIAGNOSTIC_OFFSET)
        with torch.no_grad():
            mean_code = model.codes_for_task_ids(fit_ids).mean(dim=0, keepdim=True)
        scores["linear_map_residual"] = linear_map_residual(model, batch["state"], batch["x_t"], batch["time"], mean_code)
    return scores


def code_hashes():
    hashes = _code_hashes()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        hashes[str(path.relative_to(run_loto.ROOT.parent))] = run_loto.file_sha(path)
    nl_lib = run_loto.ROOT / "lib" / "nonlinear_atom_field.py"
    hashes[str(nl_lib.relative_to(run_loto.ROOT.parent))] = run_loto.file_sha(nl_lib)
    return hashes


run_loto.build_model = build_model
run_loto.parameter_report = parameter_report
run_loto.score_code_affine = score_code_affine
run_loto.code_hashes = code_hashes
run_loto.PROTOCOL = "revision_20261024_loto_nl_v1"

if __name__ == "__main__":
    run_loto.main()
