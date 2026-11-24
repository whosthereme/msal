from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import torch

PROTOCOL = "review_20261113_action_metric_v1"
BASE = Path.home() / "vla_work" / "msal_trunkcond"
STALE_SECONDS = 2700


def read_jsonl(path):
  if not path.exists():
    return []
  return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def stored_scores(arm, seed):
  path = BASE / "stored" / "n1n2" / ("blind_nl_%s_seed%d" % (arm, seed)) / "folds.jsonl"
  return dict((int(r["held_task"]), float(r["main_selection"])) for r in read_jsonl(path))


def finished(arm, seed):
  done = set()
  for path in BASE.glob("out_*/%s_seed%d.jsonl" % (arm, seed)):
    if not path.parent.name.endswith(("_SMOKE", "dup")):
      done.update(int(r["held_task"]) for r in read_jsonl(path))
  return done


def claim(arm, seed, task, lane):
  path = BASE / "claims" / ("%s_%d_%d" % (arm, seed, task))
  try:
    os.mkdir(path)
    (path / lane).write_text(str(time.time()))
    return True
  except FileExistsError:
    pass
  if time.time() - path.stat().st_mtime < STALE_SECONDS:
    return False
  stale = path.with_name(path.name + ".stale_%s_%d" % (lane, int(time.time())))
  try:
    os.rename(path, stale)
  except OSError:
    return False
  os.mkdir(path)
  (path / lane).write_text(str(time.time()))
  return True


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--arm", required=True)
  parser.add_argument("--seeds", required=True)
  parser.add_argument("--lane", required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--smoke-steps", type=int, default=0)
  parser.add_argument("--held-tasks", default="")
  args = parser.parse_args()
  if socket.gethostname() == "workstation":
    raise RuntimeError("tc_run_any is for ws3 / ws4 only")

  import tc_arms
  from msal_ext import ideal
  from msal_ext.hyb import e2, run_loto, run_nl
  from msal_exp.data_lang import SPLIT_SELECTION, load_lang_v5
  from lib.frozen_data import DeviceCache
  from lib.pool_v5 import source_task_ids
  from lib.structured_field import channel_weights
  from run_e35_1_zero_shot_code_loto import NOISE_DRAWS

  arm, device, lane = args.arm, args.device, args.lane
  if arm not in run_nl.arms.ARMS:
    raise ValueError("unknown arm %r" % arm)
  if run_nl.arms.ARMS[arm]["family"] not in ("film", "plain"):
    raise ValueError("arm %r is not scored through forward_descriptor" % arm)
  if e2.TRAIN_STEPS != 12_800:
    raise RuntimeError("TRAIN_STEPS is not the published 12,800; unset G12_TRAIN_STEPS")
  smoke = bool(args.smoke_steps)
  if smoke:
    e2.TRAIN_STEPS = int(args.smoke_steps)
  out = BASE / (("out_q_%s" % lane) + ("_SMOKE" if smoke else ""))
  out.mkdir(parents=True, exist_ok=True)
  (BASE / "claims").mkdir(exist_ok=True)
  seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

  data = load_lang_v5("blind", verify_hash=False)
  cache = DeviceCache(data, device)
  data.state_features = None
  all_ids = [int(t) for t in source_task_ids()]
  tasks = [int(t) for t in args.held_tasks.split(",") if t.strip()] if args.held_tasks else all_ids
  weights = channel_weights(int(data.action_chunks.shape[2]), dead_dims=e2.DEAD_ACTION_DIMS, device=device)
  if arm in tc_arms.CLASSES:
    print("[any] selfcheck %s" % json.dumps(tc_arms.selfcheck(arm, run_nl.arms.config_for(arm, data), all_ids[1:], device)),
      flush=True)
  info = dict(protocol=PROTOCOL, arm=arm, seeds=seeds, lane=lane, smoke=smoke, host=socket.gethostname(),
    environment=run_loto.environment(), started_unix=time.time(), data_hash="verified by tc_run on this machine",
    code_sha256=dict((p.name, run_loto.file_sha(p)) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))),
    repo_code_sha256=run_loto.code_hashes(), sealed_rows_read=0, rollouts_executed=0)
  (out / ("%s_info_%d.json" % (arm, int(time.time())))).write_text(json.dumps(info, indent=1))

  def run_fold(seed, task, stored):
    started = time.time()
    fit_ids = [t for t in all_ids if t != task]
    model, _ = run_loto.train_arm(arm, seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
    train_seconds = time.time() - started
    model.eval()
    flow = run_loto.score_forward(model, arm, held=task, seed=seed, data=data, cache=cache, device=device)
    rows = data.rows_for_task(task, SPLIT_SELECTION)
    descriptor = torch.tensor(run_nl.arms.DESCRIPTORS["saturated"]([task]), dtype=torch.float32, device=device)
    action = ideal.action_error(model, rows, descriptor, cache, seed, weights, NOISE_DRAWS, run_loto.DRAW_STRIDE)
    record = dict(arm=arm, seed=seed, held_task=task, train_steps=int(e2.TRAIN_STEPS), lane=lane,
      flow_selection=flow["main_selection"], flow_robustness=flow["main_robustness"], action_selection=action,
      rows=int(len(rows)), stored_same_arm_flow=stored.get(task), train_seconds=train_seconds,
      seconds=time.time() - started)
    record.update(run_loto.parameter_report(model, arm))
    with (out / ("%s_seed%d.jsonl" % (arm, seed))).open("a") as handle:
      handle.write(json.dumps(record) + "\n")
    print("[any %s %s] seed=%d held=%2d %.0fs" % (arm, lane, seed, task, time.time() - started), flush=True)
    del model
    torch.cuda.empty_cache()

  def sweep_seed(seed):
    stored = stored_scores(arm, seed)
    for task in tasks:
      wanted = task not in finished(arm, seed)
      if wanted and (smoke or claim(arm, seed, task, lane)):
        run_fold(seed, task, stored)

  for sweep in ((1,) if smoke else (1, 2)):
    for seed in seeds:
      sweep_seed(seed)
  print("any %s lane %s done" % (arm, lane), flush=True)


if __name__ == "__main__":
  main()
