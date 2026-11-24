from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

PROTOCOL = "review_20261112_trunkcond_v1"
IDEAL_OUT = Path("/home/ubuntu/work/manuscript6/ideal/out")


def read_jsonl(path):
  if not path.exists():
    return []
  return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--arm", required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--held-tasks", default="")
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--smoke-steps", type=int, default=0)
  parser.add_argument("--skip-hash", action="store_true")
  args = parser.parse_args()
  if args.device == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
    raise RuntimeError("GPU runs must go through gpu_run (CUDA_VISIBLE_DEVICES=1); GPU 0 is never used")

  import tc_arms
  from msal_ext import hyb, ideal
  from msal_ext.hyb import e2, run_loto, run_nl
  from msal_exp.data_lang import SPLIT_SELECTION, load_lang_v5
  from lib.frozen_data import DeviceCache
  from lib.pool_v5 import source_task_ids
  from lib.structured_field import channel_weights
  from run_e35_1_zero_shot_code_loto import NOISE_DRAWS

  arm, seed, device = args.arm, args.seed, args.device
  if arm != "film_nl" and arm not in tc_arms.CLASSES:
    raise ValueError("arm must be film_nl (anchor) or one of %s" % sorted(tc_arms.CLASSES))
  if e2.TRAIN_STEPS != 12_800:
    raise RuntimeError("TRAIN_STEPS is not the published 12,800; unset G12_TRAIN_STEPS")
  if args.smoke_steps:
    e2.TRAIN_STEPS = int(args.smoke_steps)
    args.output = args.output.with_name(args.output.name + "_SMOKE")
  args.output.mkdir(parents=True, exist_ok=True)
  path = args.output / ("%s_seed%d.jsonl" % (arm, seed))
  done = set(int(r["held_task"]) for r in read_jsonl(path))

  data = load_lang_v5("blind", verify_hash=not args.skip_hash)
  cache = DeviceCache(data, device)
  data.state_features = None
  all_ids = [int(t) for t in source_task_ids()]
  tasks = [int(t) for t in args.held_tasks.split(",") if t.strip()] if args.held_tasks else all_ids
  weights = channel_weights(int(data.action_chunks.shape[2]), dead_dims=e2.DEAD_ACTION_DIMS, device=device)
  stored_flow = hyb.stored_film_nl(seed)
  stored_act = dict((int(r["task"]), float(r["action_selection"]))
    for r in read_jsonl(IDEAL_OUT / ("zero_seed%d.jsonl" % seed)))

  check = None
  if arm in tc_arms.CLASSES:
    check = tc_arms.selfcheck(arm, run_nl.arms.config_for(arm, data), all_ids[1:], device)
    print("[tc] selfcheck %s %s" % (arm, json.dumps(check)), flush=True)
  info = dict(protocol=PROTOCOL, arm=arm, seed=seed, smoke=bool(args.smoke_steps), selfcheck=check,
    data_sha256=data.sha256 or "not_verified", environment=run_loto.environment(),
    code_sha256=dict((p.name, run_loto.file_sha(p)) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))),
    repo_code_sha256=run_loto.code_hashes(), started_unix=time.time(), sealed_rows_read=0, rollouts_executed=0)
  info_path = args.output / ("%s_seed%d_info.json" % (arm, seed))
  if not info_path.exists():
    info_path.write_text(json.dumps(info, indent=1))

  for task in [t for t in tasks if t not in done]:
    started = time.time()
    fit_ids = [t for t in all_ids if t != task]
    model, _ = run_loto.train_arm(arm, seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
    train_seconds = time.time() - started
    model.eval()
    flow = run_loto.score_forward(model, arm, held=task, seed=seed, data=data, cache=cache, device=device)
    rows = data.rows_for_task(task, SPLIT_SELECTION)
    descriptor = torch.tensor(run_nl.arms.DESCRIPTORS["saturated"]([task]), dtype=torch.float32, device=device)
    action = ideal.action_error(model, rows, descriptor, cache, seed, weights, NOISE_DRAWS, run_loto.DRAW_STRIDE)
    record = dict(arm=arm, seed=seed, held_task=task, train_steps=int(e2.TRAIN_STEPS),
      flow_selection=flow["main_selection"], flow_robustness=flow["main_robustness"], action_selection=action,
      rows=int(len(rows)), stored_film_nl_flow=stored_flow.get(task), stored_film_nl_action=stored_act.get(task),
      train_seconds=train_seconds, seconds=time.time() - started)
    record.update(run_loto.parameter_report(model, arm))
    if arm == "film_nl":
      record["anchor_flow_bit_exact"] = stored_flow.get(task) == flow["main_selection"]
      record["anchor_action_bit_exact"] = stored_act.get(task) == action
    with path.open("a") as handle:
      handle.write(json.dumps(record) + "\n")
    print("[tc %s] seed=%d held=%2d flow=%.6f (film_nl stored %.6f) act=%.6f (stored %.6f) %.0fs" % (
      arm, seed, task, flow["main_selection"], stored_flow.get(task, float("nan")), action,
      stored_act.get(task, float("nan")), time.time() - started), flush=True)
    del model
    if device == "cuda":
      torch.cuda.empty_cache()
  print("tc %s seed %d done" % (arm, seed), flush=True)


if __name__ == "__main__":
  main()
