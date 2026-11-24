from __future__ import annotations

from pathlib import Path

from msal_exp import run_loto
from msal_exp import arms
from msal_ext.tied import TiedAtomSAL

arms.ARMS["sal_tied"] = {"family": "sal_tied", "descriptor": "saturated", "reg": True}
arms.CODE_AFFINE_FAMILIES.add("sal_tied")

_build_model = arms.build_model
_parameter_report = arms.parameter_report
_code_hashes = run_loto.code_hashes


def build_model(arm, config, fit_ids):
    if arms.ARMS[arm]["family"] == "sal_tied":
        return TiedAtomSAL(config, fit_ids, arms.DESCRIPTORS[arms.ARMS[arm]["descriptor"]])
    return _build_model(arm, config, fit_ids)


def parameter_report(model, arm):
    if arms.ARMS[arm]["family"] != "sal_tied":
        return _parameter_report(model, arm)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    conditioned = sum(p.numel() for n, p in trainable.items() if n.startswith(("atom_scale", "atom_shift", "code_map")))
    return {"trainable_parameters": int(sum(p.numel() for p in trainable.values())),
            "task_conditioned_parameters": int(conditioned)}


def code_hashes():
    hashes = _code_hashes()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        hashes[str(path.relative_to(run_loto.ROOT.parent))] = run_loto.file_sha(path)
    return hashes


run_loto.build_model = build_model
run_loto.parameter_report = parameter_report
run_loto.code_hashes = code_hashes
run_loto.PROTOCOL = "revision_20261021_loto_tied_v1"

if __name__ == "__main__":
    run_loto.main()
