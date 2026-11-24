from __future__ import annotations

import torch

from msal_ext import run_nlb
from joint_sal_velocity import JointSALConfig
from lib.pool_v5 import source_task_ids


def main() -> None:
    torch.manual_seed(0)
    ids = list(source_task_ids())[1:]
    config = JointSALConfig(state_dim=31_680, chunk_length=16, action_dim=7, k_max=16)
    model = run_nlb.run_nl.build_model("sal_nlb", config, ids).eval()
    report = run_nlb.parameter_report(model, "sal_nlb")
    state, x_t, time = torch.randn(8, 31_680), torch.randn(8, 16, 7), torch.rand(8)
    with torch.no_grad():
        z1, z2 = torch.randn(8, 16), torch.randn(8, 16)
        v = lambda z: model.forward_with_code(state, x_t, time, z)
        gap = (v(z1 + z2) - v(z1) - v(z2) + v(torch.zeros_like(z1))).abs().max().item()
        base, atoms = model.components(state, x_t, time, None)
    assert gap < 1e-4, gap
    assert base.shape == (8, 16, 7) and atoms.shape == (8, 16, 16, 7)
    assert base.abs().max() > 0, "nonlinear base should be nonzero at initialization"
    print(f"sal_nlb ok  task-conditioned {report['task_conditioned_parameters']:,}  nonlinear base "
          f"{report['nonlinear_readout_parameters']:,}  sum {report['task_conditioned_parameters'] + report['nonlinear_readout_parameters']:,}"
          f"  total {report['trainable_parameters']:,}  affinity gap {gap:.1e}")


if __name__ == "__main__":
    main()
