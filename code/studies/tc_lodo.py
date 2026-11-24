import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE), str(BASE / "trunkcond"), str(BASE / "repo" / "sal_v2_identifiable"), str(BASE / "repo")]
if sys.platform == "win32":
  import boot
import numpy as np
import torch

PROTOCOL = "review_20261113_lodo_v1"
FEATURES = dict(lodo_sat="saturated_v5", lodo_xy6="physical_xy_radial_v5", lodo_h9="harmonic2_radial2_v5",
  lodo_txy6="physical_xy_radial_v5")


def read_jsonl(path):
  if not path.exists():
    return []
  return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def descriptor(feature_set):
  from lib.task_algebra_v5 import design_matrix_for_tasks
  return lambda ids: np.asarray(design_matrix_for_tasks([int(t) for t in ids], feature_set)[0], dtype=np.float64)


def register():
  import tc_arms
  from msal_ext import run_nl
  from msal_ext.nl import FiLMNonlinearReadout, PlainNonlinearReadout

  arms = run_nl.arms

  class TrunkFiLMDesc(tc_arms.TrunkFiLMField):
    def __init__(self, config, source_task_ids):
      PlainNonlinearReadout.__init__(self, config, source_task_ids)
      matrix = descriptor(FEATURES["lodo_txy6"])(source_task_ids)
      self.register_buffer("task_features", torch.tensor(matrix, dtype=torch.float32), persistent=True)
      self.trunk_gamma = tc_arms.bias_free(int(matrix.shape[1]), config.hidden_dim)
      self.trunk_beta = tc_arms.bias_free(int(matrix.shape[1]), config.hidden_dim)

  def film_builder(fn):
    return lambda config, ids: FiLMNonlinearReadout(config, ids, fn)

  for name, feature_set in FEATURES.items():
    arms.DESCRIPTORS[name] = descriptor(feature_set)
    arms.ARMS[name] = dict(family="film", descriptor=name, reg=False)
  for name in ("lodo_sat", "lodo_xy6", "lodo_h9"):
    run_nl.BUILDERS[name] = film_builder(arms.DESCRIPTORS[name])
    run_nl.CONDITIONED[name] = ("film_gamma", "film_beta")
  run_nl.BUILDERS["lodo_txy6"] = lambda config, ids: TrunkFiLMDesc(config, ids)
  run_nl.CONDITIONED["lodo_txy6"] = ("trunk_gamma", "trunk_beta")
  arms.ARMS["lodo_plain"] = dict(family="plain", descriptor=None, reg=False)
  run_nl.BUILDERS["lodo_plain"] = lambda config, ids: PlainNonlinearReadout(config, ids)
  run_nl.CONDITIONED["lodo_plain"] = ()
  return arms


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--arm", required=True)
  parser.add_argument("--directions", default="")
  parser.add_argument("--smoke-steps", type=int, default=0)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", type=int, default=148021)
  args = parser.parse_args()

  SEED = args.seed
  arms = register()
  from msal_ext import ideal
  from msal_ext.hyb import e2, run_loto
  from msal_exp.data_lang import SPLIT_SELECTION, load_lang_v5
  from lib.frozen_data import DeviceCache
  from lib.pool_v5 import build_pool, source_task_ids
  from lib.structured_field import channel_weights
  from run_e35_1_zero_shot_code_loto import NOISE_DRAWS

  arm, device = args.arm, args.device
  assert arm in arms.ARMS and arm.startswith("lodo_"), arm
  assert e2.TRAIN_STEPS == 12_800
  smoke = bool(args.smoke_steps)
  if smoke:
    e2.TRAIN_STEPS = int(args.smoke_steps)
  out = BASE / ("out_lodo" + ("_SMOKE" if smoke else ""))
  out.mkdir(exist_ok=True)
  path = out / ("%s_seed%d.jsonl" % (arm, SEED))
  data = load_lang_v5("blind", verify_hash=False)
  cache = DeviceCache(data, device)
  data.state_features = None
  all_ids = [int(t) for t in source_task_ids()]
  direction_of = dict((t.task_id, t.direction_16) for t in build_pool())
  weights = channel_weights(int(data.action_chunks.shape[2]), dead_dims=e2.DEAD_ACTION_DIMS, device=device)
  wanted = [int(d) for d in args.directions.split(",") if d.strip()] if args.directions else list(range(16))
  info = dict(protocol=PROTOCOL, arm=arm, seed=SEED, smoke=smoke, feature_set=FEATURES.get(arm),
    environment=run_loto.environment(), repo_code_sha256=run_loto.code_hashes(), started_unix=time.time(),
    data_hash="verified by tc_run_local on this machine", sealed_rows_read=0, rollouts_executed=0)
  (out / ("%s_info_%d.json" % (arm, int(time.time())))).write_text(json.dumps(info, indent=1))
  is_film = arms.ARMS[arm]["family"] == "film"

  def score(model, task, n_fit, direction, train_seconds):
    flow = run_loto.score_forward(model, arm, held=task, seed=SEED, data=data, cache=cache, device=device)
    rows = data.rows_for_task(task, SPLIT_SELECTION)
    d = torch.tensor(arms.DESCRIPTORS[arm]([task]), dtype=torch.float32, device=device) if is_film else None
    action = lodo_action_error(model, rows, d)
    record = dict(arm=arm, seed=SEED, held_direction=direction, held_task=task, n_fit=n_fit,
      train_steps=int(e2.TRAIN_STEPS), flow_selection=flow["main_selection"],
      flow_robustness=flow["main_robustness"], action_selection=action, rows=int(len(rows)),
      train_seconds=train_seconds)
    record.update(run_loto.parameter_report(model, arm))
    with path.open("a") as handle:
      handle.write(json.dumps(record) + "\n")

  class NoDescriptor:
    def expand(self, n, _):
      return None

  def lodo_action_error(model, rows, d):
    return ideal.action_error(model, rows, d if is_film else NoDescriptor(), cache, SEED, weights, NOISE_DRAWS,
      run_loto.DRAW_STRIDE)

  def run_direction(direction):
    held = [t for t in all_ids if direction_of[t] == direction]
    fit_ids = [t for t in all_ids if direction_of[t] != direction]
    started = time.time()
    model, _ = run_loto.train_arm(arm, seed=SEED, fit_ids=fit_ids, data=data, cache=cache, device=device)
    model.eval()
    train_seconds = time.time() - started
    for task in held:
      score(model, task, len(fit_ids), direction, train_seconds)
    print("[lodo %s] direction=%2d held=%s %.0fs" % (arm, direction, held, time.time() - started), flush=True)
    del model
    torch.cuda.empty_cache()

  def pending(direction):
    done = set(int(r["held_direction"]) for r in read_jsonl(path))
    return direction not in done and (smoke or not (BASE / "STOP_LODO").exists())

  for direction in wanted:
    if pending(direction):
      run_direction(direction)
  print("lodo %s done" % arm, flush=True)


if __name__ == "__main__":
  main()
