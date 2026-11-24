from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, "/home/ubuntu/work/pilot/closed_loop")
import task_cond as TC

_STATE: dict = {"override": None}


class ModMLP(nn.Module):
    def __init__(self, f: int, hidden: int):
        super().__init__()
        self.gamma = nn.Linear(f, hidden, bias=False)
        self.beta = nn.Linear(f, hidden, bias=False)
        nn.init.normal_(self.gamma.weight, std=0.02)
        nn.init.normal_(self.beta.weight, std=0.02)
        self.current: torch.Tensor | None = None

    def modulate(self, u: torch.Tensor) -> torch.Tensor:
        if self.current is None:
            raise RuntimeError("task code not set")
        d = self.current.to(device=u.device, dtype=u.dtype)
        if d.shape[0] != u.shape[0]:
            if d.shape[0] == 1:
                d = d.expand(u.shape[0], -1)
            else:
                raise RuntimeError(f"task code batch {d.shape[0]} != {u.shape[0]}")
        return u * (1.0 + self.gamma(d)[:, None, :]) + self.beta(d)[:, None, :]


def set_override(d: torch.Tensor | None) -> None:
    _STATE["override"] = d


def install(tasks_json: str) -> "TC.TaskDescriptor":
    from lerobot.policies.smolvla import modeling_smolvla as m

    tasks = TC.load_tasks(tasks_json)
    descriptor = TC.TaskDescriptor.from_tasks(tasks)
    TC._STATE.update(kind="modmlp", descriptor=descriptor)
    orig_init, orig_fwd, orig_sample = m.VLAFlowMatching.__init__, m.VLAFlowMatching.forward, m.VLAFlowMatching.sample_actions

    def init(self, config, *args, **kwargs):
        orig_init(self, config, *args, **kwargs)
        self.task_mod = ModMLP(descriptor.dim, self.action_time_mlp_in.out_features)
        mod = self.task_mod
        self.action_time_mlp_in.register_forward_hook(lambda module, inp, out: mod.modulate(out))

    def current(self, lang_tokens, lang_masks):
        o = _STATE["override"]
        return o.reshape(1, -1) if o is not None else TC._descriptors(self, lang_tokens, lang_masks)

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
        self.task_mod.current = current(self, lang_tokens, lang_masks)
        return orig_fwd(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        self.task_mod.current = current(self, lang_tokens, lang_masks)
        return orig_sample(self, images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs)

    m.VLAFlowMatching.__init__ = init
    m.VLAFlowMatching.forward = forward
    m.VLAFlowMatching.sample_actions = sample_actions
    return descriptor
