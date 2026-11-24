import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

RES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SEEDS = (148021, 148022, 148023, 148024, 148025)
FIELDS = (("sampled chunk", "action_selection"), ("flow MSE", "flow_selection"), ("flow MSE, ROBUSTNESS rows", "flow_robustness"))


def read(path):
  if not Path(path).exists():
    return []
  return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def pct(v):
  return 100.0 * (math.exp(v) - 1.0)


NEW = {}
for p in sys.argv[3:]:
  for r in read(p):
    if r.get("train_steps") != 12800:
      continue
    k = (int(r["seed"]), r["arm"], int(r["held_task"]))
    assert k not in NEW, k
    NEW[k] = r
LODO = {}
for d in (RES / "local" / "out_lodo", RES / "final_nodes" / "out_lodo"):
  for arm in ("lodo_plain", "lodo_xy6", "lodo_sat"):
    for s in SEEDS:
      for r in read(d / ("%s_seed%d.jsonl" % (arm, s))):
        LODO.setdefault((s, arm, int(r["held_task"])), r)
SPAN = {}
for p in sorted(list((RES / "span" / "b").glob("out*.jsonl"))):
  for r in read(p):
    if r.get("train_steps") == 12800 and r["arm"] in ("lodo_xy6", "lodo_xy6id2"):
      SPAN[(int(r["seed"]), r["arm"], int(r["held_task"]))] = r
TASKS = sorted(set(k[2] for k in NEW))
missing = [(s, a, t) for s in SEEDS for a in ("lodo_sat", "lodo_xy6id2") for t in TASKS if (s, a, t) not in NEW]
print("new records", len(NEW), "tasks", len(TASKS), "missing", len(missing))
assert len(TASKS) == 41 and not missing, missing[:5]
DIR = dict((t, int(NEW[(SEEDS[0], "lodo_sat", t)]["held_direction"])) for t in TASKS)
DIRS = sorted(set(DIR.values()))
REPORT = dict(comparisons={}, checks={})


def value(source, arm, field, s, t):
  if source == "proj":
    return float(NEW[(s, arm, t)]["proj_" + field])
  table = dict(raw=NEW, lodo=LODO, span=SPAN)[source]
  return float(table[(s, arm, t)][field])


def compare(name, first, second, field):
  lr = dict((s, np.array([math.log(value(*first, field, s, t) / value(*second, field, s, t)) for t in TASKS])) for s in SEEDS)
  means = np.array([lr[s].mean() for s in SEEDS])
  h = stats.t.ppf(0.975, len(SEEDS) - 1) * means.std(ddof=1) / math.sqrt(len(SEEDS))
  per_task = np.mean([lr[s] for s in SEEDS], axis=0)
  per_dir = np.array([per_task[[i for i, t in enumerate(TASKS) if DIR[t] == d]].mean() for d in DIRS])
  b = per_dir[np.random.default_rng(0).integers(0, len(DIRS), size=(20000, len(DIRS)))].mean(axis=1)
  out = dict(pct=pct(means.mean()), lo=pct(means.mean() - h), hi=pct(means.mean() + h), per_seed=[pct(x) for x in means],
    tasks_lower=int((per_task < 0).sum()), dirs_lower=int((per_dir < 0).sum()),
    dir_boot=[pct(np.percentile(b, 2.5)), pct(np.percentile(b, 97.5))])
  out["reading"] = "lower" if out["hi"] < 0 else ("higher" if out["lo"] > 0 else "not distinguishable")
  REPORT["comparisons"].setdefault(name, {})[field] = out
  return out


PAIRS = (
  ("J1 harmonic projected vs harmonic raw", ("proj", "lodo_sat"), ("raw", "lodo_sat")),
  ("J2 harmonic projected vs no description (stored)", ("proj", "lodo_sat"), ("lodo", "lodo_plain")),
  ("J3 harmonic projected vs coordinates (stored)", ("proj", "lodo_sat"), ("lodo", "lodo_xy6")),
  ("J4 identifier projected vs identifier raw", ("proj", "lodo_xy6id2"), ("raw", "lodo_xy6id2")),
  ("J5 identifier projected vs coordinates alone (stored)", ("proj", "lodo_xy6id2"), ("span", "lodo_xy6")),
  ("context: harmonic raw (this run) vs no description (stored)", ("raw", "lodo_sat"), ("lodo", "lodo_plain")),
  ("context: identifier raw (this run) vs coordinates alone (stored)", ("raw", "lodo_xy6id2"), ("span", "lodo_xy6")),
  ("check: harmonic raw (this run) vs harmonic raw (stored)", ("raw", "lodo_sat"), ("lodo", "lodo_sat")),
  ("check: identifier raw (this run) vs identifier raw (stored)", ("raw", "lodo_xy6id2"), ("span", "lodo_xy6id2")),
)
for name, first, second in PAIRS:
  print("==", name)
  for label, field in FIELDS:
    r = compare(name, first, second, field)
    print("  %-26s %+7.1f%% [%+7.1f, %+7.1f]  %-19s per seed %s | tasks lower %d/41, directions lower %d/16, direction bootstrap %+.1f..%+.1f"
      % (label, r["pct"], r["lo"], r["hi"], r["reading"], " ".join("%+.1f" % x for x in r["per_seed"]),
         r["tasks_lower"], r["dirs_lower"], r["dir_boot"][0], r["dir_boot"][1]))

for arm, table in (("lodo_sat", LODO), ("lodo_xy6id2", SPAN)):
  keys = [(s, arm, t) for s in SEEDS for t in TASKS]
  exact = sum(NEW[k]["flow_selection"] == table[k]["flow_selection"] and NEW[k]["action_selection"] == table[k]["action_selection"] for k in keys)
  scorer = sum(bool(NEW[k]["raw_matches_stored_scorer"]) for k in keys)
  iota = np.array([NEW[(SEEDS[0], arm, t)]["iota"] for t in TASKS])
  ranks = sorted(set(int(NEW[k]["span_rank"]) for k in keys))
  REPORT["checks"][arm] = dict(raw_bit_exact_with_stored=int(exact), of=len(keys), scorer_self_check=int(scorer),
    iota_mean=float(iota.mean()), iota_min=float(iota.min()), iota_max=float(iota.max()), span_ranks=ranks,
    descriptor_dim=int(NEW[keys[0]]["descriptor_dim"]),
    norm_ratio_mean=float(np.mean([NEW[(SEEDS[0], arm, t)]["projected_norm"] / NEW[(SEEDS[0], arm, t)]["descriptor_norm"] for t in TASKS])))
  print("check %-12s raw scores bit-exact with stored %d/%d; scorer self-check %d/%d; span share mean %.3f (%.3f to %.3f); span rank %s of %d columns"
    % (arm, exact, len(keys), scorer, len(keys), iota.mean(), iota.min(), iota.max(), ranks, REPORT["checks"][arm]["descriptor_dim"]))

j1 = REPORT["comparisons"]["J1 harmonic projected vs harmonic raw"]["flow_selection"]["reading"]
j4 = REPORT["comparisons"]["J4 identifier projected vs identifier raw"]["action_selection"]["reading"]
j2 = REPORT["comparisons"]["J2 harmonic projected vs no description (stored)"]["action_selection"]["reading"]
rule = dict(lower="supported", higher="contradicted")
REPORT["readings"] = dict(
  out_of_span_harms_harmonic=rule.get(j1, "not supported"), out_of_span_harms_identifier=rule.get(j4, "not supported"),
  in_span_part_of_harmonic_usable="supported" if j2 == "lower" else "not shown")
print("readings:", json.dumps(REPORT["readings"]))
OUT.write_text(json.dumps(REPORT, indent=1))
print("written", OUT)
