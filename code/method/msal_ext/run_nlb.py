from __future__ import annotations

from pathlib import Path

from msal_ext import run_nl
from msal_ext.nlb import NonlinearBaseSAL

run_nl.arms.ARMS["sal_nlb"] = {"family": "sal_v5", "descriptor": "saturated", "reg": True}
run_nl.BUILDERS["sal_nlb"] = lambda config, ids: NonlinearBaseSAL(config, ids)
run_nl.CONDITIONED["sal_nlb"] = ("atom_out", "code_map")

_parameter_report = run_nl.parameter_report


def parameter_report(model, arm):
    report = _parameter_report(model, arm)
    if arm == "sal_nlb":
        report["nonlinear_readout_parameters"] = int(sum(p.numel() for n, p in model.named_parameters()
                                                         if n.startswith("base_out.mlp") and p.requires_grad))
    return report


_code_hashes = run_nl.code_hashes


def code_hashes():
    hashes = _code_hashes()
    path = Path(__file__).resolve().parent / "nlb.py"
    hashes[str(path.relative_to(run_nl.run_loto.ROOT.parent))] = run_nl.run_loto.file_sha(path)
    return hashes


run_nl.run_loto.parameter_report = parameter_report
run_nl.run_loto.code_hashes = code_hashes
run_nl.run_loto.PROTOCOL = "revision_20261024_loto_nl_exploratory_nlb_v1"

if __name__ == "__main__":
    run_nl.run_loto.main()
