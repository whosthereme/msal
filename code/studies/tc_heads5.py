import argparse
import json
import socket
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE), str(BASE / "trunkcond"), str(BASE / "repo" / "sal_v2_identifiable"), str(BASE / "repo")]
if sys.platform == "win32":
  import boot
import numpy as np
import torch

PROTOCOL = "review_20261118_heads5_v1"
GROUPS = dict(a=("film_nl", "plain_nl", "film", "plain", "tin_nl"),
  b=("sal_nlb", "sal_nl", "moe_desc", "sal", "ctx_add", "bilinear"))


def read_jsonl(path):
  if not path.exists():
    return []
  return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def stored_flow(n1n2, tc_out, seed):
  out = {}
  if n1n2 and Path(n1n2).exists():
    for path in sorted(Path(n1n2).glob("*seed%d/folds.jsonl" % seed)):
      if "anchor" in path.parent.name or path.parent.name.startswith("lang"):
        continue
      for r in read_jsonl(path):
        if r.get("feature") == "blind":
          out.setdefault(r["arm"], {})[int(r["held_task"])] = float(r["main_selection"])
  if tc_out and Path(tc_out).exists():
    for r in read_jsonl(Path(tc_out) / ("tin_nl_seed%d.jsonl" % seed)):
      out.setdefault("tin_nl", {})[int(r["held_task"])] = float(r["flow_selection"])
  return out


def stored_action(ideal_out, tc_out, seed):
  out = {}
  if ideal_out and Path(ideal_out).exists():
    rows = read_jsonl(Path(ideal_out) / ("zero_seed%d.jsonl" % seed))
    out["film_nl"] = dict((int(r["task"]), float(r["action_selection"])) for r in rows)
  if tc_out and Path(tc_out).exists():
    rows = read_jsonl(Path(tc_out) / ("tin_nl_seed%d.jsonl" % seed))
    out["tin_nl"] = dict((int(r["held_task"]), float(r["action_selection"])) for r in rows)
  return out


class NoDescriptor:
  def expand(self, n, _):
    return None


class Field:

  def __init__(self, fn):
    self.fn = fn

  def forward_descriptor(self, state, x, t, d):
    return self.fn(state, x, t)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--held-tasks", required=True)
  parser.add_argument("--group", choices=sorted(GROUPS), required=True)
  parser.add_argument("--arms", default="")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--stored-n1n2", default="")
  parser.add_argument("--stored-ideal", default="")
  parser.add_argument("--stored-tc", default="")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--smoke-steps", type=int, default=0)
  parser.add_argument("--lane", default="")
  args = parser.parse_args()

  import tc_arms
  from msal_ext import ideal, run_nlb
  from msal_ext.hyb import e2, run_loto, run_nl
  from msal_exp.arms import CODE_AFFINE_FAMILIES
  from msal_exp.data_lang import SPLIT_ROBUSTNESS, SPLIT_SELECTION, load_lang_v5
  from lib.frozen_data import DeviceCache
  from lib.pool_v5 import source_task_ids
  from lib.structured_field import channel_weights
  from run_e35_1_zero_shot_code_loto import NOISE_DRAWS, collect_blocks, weighted_mse

  ARMS = run_nl.arms.ARMS
  SEED, device = args.seed, args.device
  run_arms = [a for a in args.arms.split(",") if a] if args.arms else list(GROUPS[args.group])
  assert all(a in ARMS for a in run_arms), run_arms
  assert all(a in GROUPS["a"] + GROUPS["b"] for a in run_arms), run_arms
  assert e2.TRAIN_STEPS == 12_800
  if args.smoke_steps:
    e2.TRAIN_STEPS = int(args.smoke_steps)
    args.output = args.output.with_name(args.output.name + "_SMOKE")
  args.output.mkdir(parents=True, exist_ok=True)
  path = args.output / "folds.jsonl"
  done = set((int(r["seed"]), r["arm"], int(r["held_task"])) for r in read_jsonl(path))
  data = load_lang_v5("blind", verify_hash=False)
  cache = DeviceCache(data, device)
  data.state_features = None
  all_ids = [int(t) for t in source_task_ids()]
  weights = channel_weights(int(data.action_chunks.shape[2]), dead_dims=e2.DEAD_ACTION_DIMS, device=device)
  ref_flow = stored_flow(args.stored_n1n2, args.stored_tc, SEED)
  ref_act = stored_action(args.stored_ideal, args.stored_tc, SEED)
  host = socket.gethostname()
  info = dict(protocol=PROTOCOL, seed=SEED, group=args.group, arms=run_arms, smoke=bool(args.smoke_steps),
    lane=args.lane, environment=run_loto.environment(), repo_code_sha256=run_loto.code_hashes(),
    runner_sha256=run_loto.file_sha(Path(__file__).resolve()), started_unix=time.time())
  (args.output / ("run_info_%d.json" % int(time.time()))).write_text(json.dumps(info, indent=1))

  def code_affine_flow(model, task, phi):
    out = {}
    for name, split, offset in (("selection", SPLIT_SELECTION, run_loto.SELECTION_OFFSET),
        ("robustness", SPLIT_ROBUSTNESS, run_loto.ROBUSTNESS_OFFSET)):
      blocks = collect_blocks(model, data, cache, task, split=split, seed=SEED + offset, draws=NOISE_DRAWS)
      out[name] = float(np.mean([weighted_mse(b, phi, weights) for b in blocks]))
    return out

  def score(model, arm, task):
    family = ARMS[arm]["family"]
    rows = data.rows_for_task(task, SPLIT_SELECTION)
    if family in CODE_AFFINE_FAMILIES:
      with torch.no_grad():
        phi = model.codes_for_task_ids([task])[0].detach()
      flow = code_affine_flow(model, task, phi)

      def velocity(state, x, t):
        base, atoms = model.components(state, x, t, None)
        return base + torch.einsum("k,bktd->btd", phi, atoms)

      field, d = Field(velocity), NoDescriptor()
    else:
      scores = run_loto.score_forward(model, arm, held=task, seed=SEED, data=data, cache=cache, device=device)
      flow = dict(selection=scores["main_selection"], robustness=scores["main_robustness"])
      field, d = fields_forward(model, arm, family, task)
    action = ideal.action_error(field, rows, d, cache, SEED, weights, NOISE_DRAWS, run_loto.DRAW_STRIDE)
    return flow, action, int(len(rows))

  def fields_forward(model, arm, family, task):
    if family == "film":
      matrix = run_nl.arms.DESCRIPTORS[ARMS[arm]["descriptor"]]([task])
      return model, torch.tensor(matrix, dtype=torch.float32, device=device)
    if family == "plain":
      return Field(lambda state, x, t: model.forward_descriptor(state, x, t)), NoDescriptor()
    if family == "context":
      held_row = torch.tensor([model.task_features.shape[0] - 1], device=device, dtype=torch.long)
      return Field(lambda state, x, t: model.forward_with_rows(state, x, t, held_row.expand(state.shape[0]))), NoDescriptor()
    raise ValueError("no sampled-chunk path for family %r" % family)

  def run_fold(arm, task):
    started = time.time()
    fit_ids = [t for t in all_ids if t != task]
    model, _ = run_loto.train_arm(arm, seed=SEED, fit_ids=fit_ids, data=data, cache=cache, device=device)
    train_seconds = time.time() - started
    model.eval()
    flow, action, n_rows = score(model, arm, task)
    record = dict(arm=arm, seed=SEED, held_task=task, group=args.group, lane=args.lane, host=host,
      train_steps=int(e2.TRAIN_STEPS), flow_selection=flow["selection"], flow_robustness=flow["robustness"],
      action_selection=action, rows=n_rows, train_seconds=train_seconds, protocol=PROTOCOL)
    record.update(run_loto.parameter_report(model, arm))
    sf, sa = ref_flow.get(arm, {}).get(task), ref_act.get(arm, {}).get(task)
    record.update(stored_flow=sf, stored_action=sa)
    record["flow_bit_exact"] = None if sf is None else bool(sf == flow["selection"])
    record["action_bit_exact"] = None if sa is None else bool(sa == action)
    record["seconds"] = time.time() - started
    del model
    torch.cuda.empty_cache()
    return record

  for task in [int(t) for t in args.held_tasks.split(",") if t.strip()]:
    for arm in run_arms:
      if (SEED, arm, task) in done:
        continue
      if not args.smoke_steps and (BASE / "STOP_HEADS5").exists():
        print("STOP_HEADS5 present", flush=True)
        return
      r = run_fold(arm, task)
      with path.open("a") as handle:
        handle.write(json.dumps(r) + "\n")
      print("[heads5 %s] seed=%d held=%2d flow=%.6f (stored %s, exact %s) act=%.6f (exact %s) %.0fs" % (arm, SEED,
        task, r["flow_selection"], r["stored_flow"], r["flow_bit_exact"], r["action_selection"],
        r["action_bit_exact"], r["seconds"]), flush=True)
  print("heads5 unit done", flush=True)


if __name__ == "__main__":
  main()
