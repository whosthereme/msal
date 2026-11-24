import argparse
import json
import socket
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE)]
import tc_combo
import numpy as np
import torch

import tc_arms
from msal_exp.arms import _trunk_hidden
from msal_ext import hyb, ideal, run_nl
from msal_ext.hyb import e2, run_loto
from msal_ext.nl import PlainNonlinearReadout
from lib.frozen_data import DeviceCache
from lib.structured_field import channel_weights
from run_e35_1_zero_shot_code_loto import NOISE_DRAWS

PROTOCOL = "review_20261119_tin_v1"
_stored_hidden_at, _stored_invert_t = tc_combo.hidden_at, tc_combo.invert_t


def first_pre(R, state, x, t):
  return R.trunk[0](tc_arms.trunk_inputs(R, state, x, t))


def hidden_at(R, arm, state, x, t, d):
  if hasattr(R, "desc_in"):
    return tc_arms.trunk_tail(R, R.trunk[1](first_pre(R, state, x, t) + R.desc_in(d)))
  if arm == "film_nl":
    return _trunk_hidden(R, state, x, t)
  return _stored_hidden_at(R, arm, state, x, t, d)


def invert_t(R, blocks, d0, weights):
  if not hasattr(R, "desc_in"):
    return _stored_invert_t(R, blocks, d0, weights)
  pres = []
  with torch.no_grad():
    for b in blocks:
      x = b["batch"]
      pres.append(first_pre(R, x["state"], x["x_t"], x["time"]))
  d = d0.clone().requires_grad_(True)
  optimizer = torch.optim.Adam([d], lr=hyb.INVERSION_LR)
  for _ in range(hyb.INVERSION_STEPS):
    loss = 0.0
    for b, pre in zip(blocks, pres):
      n = pre.shape[0]
      hidden = tc_arms.trunk_tail(R, R.trunk[1](pre + R.desc_in(d.expand(n, -1))))
      prediction = tc_arms.shape_out(R, R.plain_head(hidden) + R.plain_mlp(hidden), n)
      loss = loss + e2.masked_channel_loss(prediction, b["batch"]["target"], b["batch"]["masks"], weights)
    optimizer.zero_grad(set_to_none=True)
    (loss / len(blocks)).backward()
    optimizer.step()
  return d.detach()


tc_combo.hidden_at, tc_combo.invert_t = hidden_at, invert_t


def startup_check(arm, data, fit_ids, device):
  from msal_exp.arms import config_for
  torch.manual_seed(1)
  model = run_nl.build_model(arm, config_for(arm, data), fit_ids).to(device).eval()
  g = torch.Generator(device="cpu").manual_seed(7)
  state = torch.randn(6, model.config.state_dim, generator=g).to(device)
  x = torch.randn(6, model.config.chunk_length, model.config.action_dim, generator=g).to(device)
  t = torch.rand(6, generator=g).to(device)
  d = model.task_features[:6]
  with torch.no_grad():
    own = model.forward_descriptor(state, x, t, d)
    hidden = tc_combo.hidden_at(model, "tin_nl", state, x, t, d)
    mine = tc_arms.shape_out(model, model.plain_head(hidden) + model.plain_mlp(hidden), 6)
    off = model.forward_descriptor(state, x, t, torch.zeros_like(d))
  gap, effect = float((own - mine).abs().max()), float((own - off).abs().max())
  if gap > 1e-5 or effect == 0.0:
    raise RuntimeError("startup check failed for %s: gap %.3e, description effect %.3e" % (arm, gap, effect))
  del model
  return dict(hidden_gap=gap, description_effect_at_init=effect)


class Field:

  def __init__(self, model):
    self.model = model

  def forward_descriptor(self, state, x, t, d):
    return self.model.forward_descriptor(state, x, t)


class NoDescriptor:
  def expand(self, n, _):
    return None


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--family", choices=("dev", "l40"), required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--held-tasks", required=True)
  parser.add_argument("--kinds", default="tin_nl")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--smoke-steps", type=int, default=0)
  parser.add_argument("--lane", default="")
  args = parser.parse_args()
  kinds = [k for k in args.kinds.split(",") if k]
  assert all(k in ("tin_nl", "plain_nl", "film_nl") for k in kinds), kinds
  assert e2.TRAIN_STEPS == 12_800
  if args.smoke_steps:
    e2.TRAIN_STEPS = int(args.smoke_steps)
    args.output = args.output.with_name(args.output.name + "_SMOKE")
  args.output.mkdir(parents=True, exist_ok=True)
  path = args.output / "folds.jsonl"
  done = set()
  if path.exists():
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    done = set((int(r["seed"]), r["arm"], int(r["held_task"])) for r in rows)
  device, seed, host = args.device, args.seed, socket.gethostname()

  if args.family == "dev":
    from msal_exp.data_lang import LANG_NPZ, load_lang_v5
    assert "plain_nl" not in kinds, "plain_nl is the LIBERO zero-shot reference only"
    tc_combo.SEED = seed
    data = load_lang_v5("blind", verify_hash=False)
    episodes = np.load(LANG_NPZ, mmap_mode="r")["episode_indices"].astype(np.int64)
    cache = DeviceCache(data, device)
    data.state_features = None
    weights = channel_weights(int(data.action_chunks.shape[2]), dead_dims=e2.DEAD_ACTION_DIMS, device=device)

    def fold(kind, held):
      return tc_combo.run_fold(kind, held, data, cache, episodes, device, weights)
  else:
    import tc_l40
    from msal_ext import l40
    from lib import g39_data
    assert e2.DEAD_ACTION_DIMS == ()

    class TrunkInputWords(tc_arms.TrunkInputField):

      def __init__(self, config, source_task_ids):
        PlainNonlinearReadout.__init__(self, config, source_task_ids)
        matrix = l40.word_descriptor(source_task_ids)
        self.register_buffer("task_features", torch.tensor(matrix, dtype=torch.float32), persistent=True)
        self.desc_in = tc_arms.bias_free(int(matrix.shape[1]), config.hidden_dim)

    run_nl.arms.ARMS["l40_tin_nl"] = dict(family="film", descriptor="l40_words", reg=False)
    run_nl.BUILDERS["l40_tin_nl"] = lambda config, ids: TrunkInputWords(config, ids)
    run_nl.CONDITIONED["l40_tin_nl"] = ("desc_in",)
    data = g39_data.load_g39_data(verify_hash=False)
    cache = DeviceCache(data, device)
    data.state_features = np.empty((0, g39_data.FROZEN_FEATURE_DIM), dtype=np.float16)
    weights = channel_weights(int(data.action_chunks.shape[2]), dead_dims=e2.DEAD_ACTION_DIMS, device=device)

    def plain_fold(held):
      started = time.time()
      fit_ids = [t for t in sorted(l40.INSTRUCTIONS) if t != held]
      R, _ = run_loto.train_arm("l40_plain_nl", seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
      R.eval()
      flow = run_loto.score_forward(R, "l40_plain_nl", held=held, seed=seed, data=data, cache=cache, device=device)
      val_rows = data.rows_for_task(held, g39_data.SPLIT_VAL)
      act = ideal.action_error(Field(R), val_rows, NoDescriptor(), cache, seed, weights, NOISE_DRAWS,
        run_loto.DRAW_STRIDE)
      rec = dict(arm="plain_nl", held_task=held, seed=seed, score_forward=dict(selection=flow["main_selection"],
        robustness=flow["main_robustness"]), action=dict(zero_shot=act), seconds=time.time() - started)
      del R
      torch.cuda.empty_cache()
      return rec

    def fold(kind, held):
      if kind == "plain_nl":
        return plain_fold(held)
      return tc_l40.run_fold(kind, held, seed, data, cache, device, weights)

  check = None
  if "tin_nl" in kinds:
    if args.family == "dev":
      from lib.pool_v5 import source_task_ids
      check = startup_check("tin_nl", data, [int(t) for t in source_task_ids()][:8], device)
    else:
      check = startup_check("l40_tin_nl", data, sorted(l40.INSTRUCTIONS)[:8], device)
  info = dict(protocol=PROTOCOL, family=args.family, seed=seed, kinds=kinds, smoke=bool(args.smoke_steps),
    lane=args.lane, host=host, startup_check=check, environment=run_loto.environment(),
    repo_code_sha256=run_loto.code_hashes(),
    runner_sha256=run_loto.file_sha(Path(__file__).resolve()), started_unix=time.time())
  (args.output / ("run_info_%d.json" % int(time.time()))).write_text(json.dumps(info, indent=1))
  for held in [int(t) for t in args.held_tasks.split(",") if t.strip()]:
    for kind in kinds:
      if (seed, kind, held) in done:
        continue
      if not args.smoke_steps and (BASE / "STOP_TIN").exists():
        print("STOP_TIN present", flush=True)
        return
      try:
        record = fold(kind, held)
      except FloatingPointError:
        record = fold(kind, held)
        record["retrained_after_nan"] = True
      record.update(lane=args.lane, host=host, family=args.family, protocol=PROTOCOL)
      with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
      print("[tin %s %s] seed=%d held=%d %.0fs" % (args.family, kind, seed, held, record["seconds"]), flush=True)
  print("tin unit done", flush=True)


if __name__ == "__main__":
  main()
