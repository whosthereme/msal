import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE), str(BASE / "trunkcond"), str(BASE / "repo" / "sal_v2_identifiable"), str(BASE / "repo")]
import boot
import torch

PROTOCOL = "review_20261113_action_metric_v1_local"


def read_jsonl(path):
  if not path.exists():
    return []
  return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def skipped(path, arm, seed):
  if not path or not Path(path).exists():
    return set()
  rows = [x.split() for x in Path(path).read_text().splitlines() if x.strip()]
  return set(int(r[2]) for r in rows if r[0] == arm and int(r[1]) == seed)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--arm", required=True)
  parser.add_argument("--seeds", required=True)
  parser.add_argument("--tasks", default="")
  parser.add_argument("--skip-file", default="")
  parser.add_argument("--smoke-steps", type=int, default=0)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()

  import tc_arms
  from msal_ext import ideal
  from msal_ext.hyb import e2, run_loto, run_nl
  from msal_exp.data_lang import SPLIT_SELECTION, load_lang_v5
  from lib.frozen_data import DeviceCache
  from lib.pool_v5 import source_task_ids
  from lib.structured_field import channel_weights
  from run_e35_1_zero_shot_code_loto import NOISE_DRAWS

  arm, device = args.arm, args.device
  assert run_nl.arms.ARMS[arm]["family"] in ("film", "plain"), arm
  assert e2.TRAIN_STEPS == 12_800
  smoke = bool(args.smoke_steps)
  if smoke:
    e2.TRAIN_STEPS = int(args.smoke_steps)
  out = BASE / ("out_q_local" + ("_SMOKE" if smoke else ""))
  out.mkdir(exist_ok=True)
  data = load_lang_v5("blind", verify_hash=False)
  cache = DeviceCache(data, device)
  data.state_features = None
  all_ids = [int(t) for t in source_task_ids()]
  tasks = [int(t) for t in args.tasks.split(",") if t.strip()] if args.tasks else all_ids
  weights = channel_weights(int(data.action_chunks.shape[2]), dead_dims=e2.DEAD_ACTION_DIMS, device=device)
  if arm in tc_arms.CLASSES:
    print("[local] selfcheck %s" % json.dumps(tc_arms.selfcheck(arm, run_nl.arms.config_for(arm, data), all_ids[1:], device)),
      flush=True)
  info = dict(protocol=PROTOCOL, arm=arm, seeds=args.seeds, smoke=smoke, environment=run_loto.environment(),
    data_hash="verified by tc_run_local on this machine", started_unix=time.time(),
    repo_code_sha256=run_loto.code_hashes(), sealed_rows_read=0, rollouts_executed=0)
  (out / ("%s_info_%d.json" % (arm, int(time.time())))).write_text(json.dumps(info, indent=1))

  def run_fold(seed, task):
    started = time.time()
    fit_ids = [t for t in all_ids if t != task]
    model, _ = run_loto.train_arm(arm, seed=seed, fit_ids=fit_ids, data=data, cache=cache, device=device)
    train_seconds = time.time() - started
    model.eval()
    flow = run_loto.score_forward(model, arm, held=task, seed=seed, data=data, cache=cache, device=device)
    rows = data.rows_for_task(task, SPLIT_SELECTION)
    descriptor = torch.tensor(run_nl.arms.DESCRIPTORS["saturated"]([task]), dtype=torch.float32, device=device)
    action = ideal.action_error(model, rows, descriptor, cache, seed, weights, NOISE_DRAWS, run_loto.DRAW_STRIDE)
    record = dict(arm=arm, seed=seed, held_task=task, train_steps=int(e2.TRAIN_STEPS), lane="local",
      flow_selection=flow["main_selection"], flow_robustness=flow["main_robustness"], action_selection=action,
      rows=int(len(rows)), train_seconds=train_seconds, seconds=time.time() - started)
    record.update(run_loto.parameter_report(model, arm))
    with (out / ("%s_seed%d.jsonl" % (arm, seed))).open("a") as handle:
      handle.write(json.dumps(record) + "\n")
    print("[local %s] seed=%d held=%2d %.0fs" % (arm, seed, task, time.time() - started), flush=True)
    del model
    torch.cuda.empty_cache()

  def wanted(seed, task):
    have = set(int(r["held_task"]) for r in read_jsonl(out / ("%s_seed%d.jsonl" % (arm, seed))))
    return task not in have and task not in skipped(args.skip_file, arm, seed) and not (BASE / "STOP").exists()

  pairs = [(int(s), t) for s in args.seeds.split(",") if s.strip() for t in tasks]
  for seed, task in pairs:
    if wanted(seed, task):
      run_fold(seed, task)
  print("local %s done" % arm, flush=True)


if __name__ == "__main__":
  main()
