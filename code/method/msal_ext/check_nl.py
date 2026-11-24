from __future__ import annotations

import torch

from msal_ext import run_nl
from msal_exp import arms
from joint_sal_velocity import JointSALConfig
from lib.pool_v5 import source_task_ids

EXPECTED_CONDITIONED = {"sal_nl": 922_128, "sal_nl_noreg": 922_128, "film_nl": 21_504, "plain_nl": 0,
                        "moe_desc": 922_128}


def main() -> None:
    torch.manual_seed(0)
    ids = list(source_task_ids())[1:]
    config = JointSALConfig(state_dim=31_680, chunk_length=16, action_dim=7, k_max=16)
    batch = 8
    state, x_t, time = torch.randn(batch, 31_680), torch.randn(batch, 16, 7), torch.rand(batch)
    rows = torch.arange(batch) % len(ids)
    for arm in EXPECTED_CONDITIONED:
        model = run_nl.build_model(arm, config, ids).eval()
        report = run_nl.parameter_report(model, arm)
        assert report["task_conditioned_parameters"] == EXPECTED_CONDITIONED[arm], (arm, report)
        with torch.no_grad():
            if arms.ARMS[arm]["family"] in arms.CODE_AFFINE_FAMILIES:
                base, atoms = model.components(state, x_t, time, None)
                out = base + torch.einsum("bk,bktd->btd", model.codes_for_rows(rows), atoms)
                z1, z2 = torch.randn(batch, 16), torch.randn(batch, 16)
                v = lambda z: model.forward_with_code(state, x_t, time, z)
                gap = (v(z1 + z2) - v(z1) - v(z2) + v(torch.zeros_like(z1))).abs().max().item()
                assert gap < 1e-4, (arm, gap)
                affinity = f"affinity gap {gap:.1e}"
            else:
                out = model.forward_with_rows(state, x_t, time, rows)
                affinity = "not code-affine (scored by forward path)"
        assert out.shape == (batch, 16, 7), (arm, tuple(out.shape))
        assert torch.isfinite(out).all(), arm
        print(f"{arm:13s} ok  task-conditioned {report['task_conditioned_parameters']:>9,}  readout "
              f"{report['nonlinear_readout_parameters']:>7,}  total {report['trainable_parameters']:>10,}  {affinity}")


if __name__ == "__main__":
    main()
