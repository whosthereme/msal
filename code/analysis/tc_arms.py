from __future__ import annotations

import torch
from torch import nn

from msal_ext import run_nl
from msal_ext.nl import FiLMNonlinearReadout, PlainNonlinearReadout
from joint_sal_velocity import sinusoidal_time_embedding

arms = run_nl.arms
SATURATED = arms.DESCRIPTORS["saturated"]
INIT_STD = 0.02


def trunk_inputs(model, state, noisy_actions, time):
  n = state.shape[0]
  normalized = (state.float() - model.state_mean) / model.state_std
  state_hidden = model.state_encoder(normalized)
  action_hidden = model.action_encoder(noisy_actions.reshape(n, -1).float())
  time_hidden = sinusoidal_time_embedding(time.float(), model.config.time_dim)
  return torch.cat([state_hidden, action_hidden, time_hidden], dim=-1)


def trunk_tail(model, first):
  return model.trunk[4](model.trunk[3](model.trunk[2](first)))


def bias_free(width, hidden):
  layer = nn.Linear(width, hidden, bias=False)
  nn.init.normal_(layer.weight, std=INIT_STD)
  return layer


def shape_out(model, out, n):
  return out.reshape(n, model.config.chunk_length, model.config.action_dim)


class TrunkInputField(PlainNonlinearReadout):

  def __init__(self, config, source_task_ids):
    super().__init__(config, source_task_ids)
    matrix = SATURATED(source_task_ids)
    self.register_buffer("task_features", torch.tensor(matrix, dtype=torch.float32), persistent=True)
    self.desc_in = bias_free(int(matrix.shape[1]), config.hidden_dim)

  def forward_descriptor(self, state, noisy_actions, time, descriptor):
    pre = self.trunk[0](trunk_inputs(self, state, noisy_actions, time)) + self.desc_in(descriptor)
    hidden = trunk_tail(self, self.trunk[1](pre))
    return shape_out(self, self.plain_head(hidden) + self.plain_mlp(hidden), state.shape[0])

  def forward_with_rows(self, state, noisy_actions, time, task_rows):
    return self.forward_descriptor(state, noisy_actions, time, self.task_features[task_rows.long()])


class TrunkFiLMField(PlainNonlinearReadout):

  def __init__(self, config, source_task_ids):
    super().__init__(config, source_task_ids)
    matrix = SATURATED(source_task_ids)
    self.register_buffer("task_features", torch.tensor(matrix, dtype=torch.float32), persistent=True)
    self.trunk_gamma = bias_free(int(matrix.shape[1]), config.hidden_dim)
    self.trunk_beta = bias_free(int(matrix.shape[1]), config.hidden_dim)

  def forward_descriptor(self, state, noisy_actions, time, descriptor):
    first = self.trunk[1](self.trunk[0](trunk_inputs(self, state, noisy_actions, time)))
    first = first * (1.0 + self.trunk_gamma(descriptor)) + self.trunk_beta(descriptor)
    hidden = trunk_tail(self, first)
    return shape_out(self, self.plain_head(hidden) + self.plain_mlp(hidden), state.shape[0])

  def forward_with_rows(self, state, noisy_actions, time, task_rows):
    return self.forward_descriptor(state, noisy_actions, time, self.task_features[task_rows.long()])


class TrunkAndOutputFiLM(FiLMNonlinearReadout):

  def __init__(self, config, source_task_ids):
    super().__init__(config, source_task_ids, SATURATED)
    width = int(self.task_features.shape[1])
    self.trunk_gamma = bias_free(width, config.hidden_dim)
    self.trunk_beta = bias_free(width, config.hidden_dim)

  def forward_descriptor(self, state, noisy_actions, time, descriptor):
    first = self.trunk[1](self.trunk[0](trunk_inputs(self, state, noisy_actions, time)))
    first = first * (1.0 + self.trunk_gamma(descriptor)) + self.trunk_beta(descriptor)
    hidden = trunk_tail(self, first)
    hidden = hidden * (1.0 + self.film_gamma(descriptor)) + self.film_beta(descriptor)
    return shape_out(self, self.film_head(hidden) + self.film_mlp(hidden), state.shape[0])


CLASSES = dict(tin_nl=TrunkInputField, tfilm_nl=TrunkFiLMField, tboth_nl=TrunkAndOutputFiLM)
NEW_WEIGHTS = dict(tin_nl=("desc_in",), tfilm_nl=("trunk_gamma", "trunk_beta"), tboth_nl=("trunk_gamma", "trunk_beta"))
CONDITIONED = dict(tin_nl=("desc_in",), tfilm_nl=("trunk_gamma", "trunk_beta"),
  tboth_nl=("trunk_gamma", "trunk_beta", "film_gamma", "film_beta"))

for _name, _cls in CLASSES.items():
  arms.ARMS[_name] = dict(family="film", descriptor="saturated", reg=False)
  run_nl.BUILDERS[_name] = (lambda cls: (lambda config, ids: cls(config, ids)))(_cls)
  run_nl.CONDITIONED[_name] = CONDITIONED[_name]


def selfcheck(arm, config, fit_ids, device):
  torch.manual_seed(12345)
  model = CLASSES[arm](config, fit_ids).to(device).eval()
  generator = torch.Generator(device="cpu").manual_seed(777)
  state = torch.randn(6, config.state_dim, generator=generator).to(device)
  x = torch.randn(6, config.chunk_length, config.action_dim, generator=generator).to(device)
  t = torch.rand(6, generator=generator).to(device)
  d = model.task_features[:6]
  with torch.no_grad():
    live = model.forward_descriptor(state, x, t, d)
    for prefix in NEW_WEIGHTS[arm]:
      getattr(model, prefix).weight.zero_()
    off = model.forward_descriptor(state, x, t, d)
    if arm == "tboth_nl":
      parent = FiLMNonlinearReadout.forward_descriptor(model, state, x, t, d)
    else:
      parent = PlainNonlinearReadout.forward_descriptor(model, state, x, t)
  gap = float((off - parent).abs().max())
  moved = float((live - off).abs().max())
  if gap > 1e-6:
    raise RuntimeError("selfcheck failed for %s: max abs difference to the parent forward pass %.3e" % (arm, gap))
  if moved == 0.0:
    raise RuntimeError("selfcheck failed for %s: the description does not reach the output" % arm)
  return dict(parent_gap=gap, description_effect_at_init=moved)
