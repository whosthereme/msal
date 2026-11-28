# MSAL: zero-shot generalization of frozen robot policies from task descriptions

Project page: https://whosthereme.github.io/msal/

This repository holds the material behind the paper *Interpretable Zero-Shot Generalization of Frozen Robot
Policies via Trunk Modulation and Span Analysis*: code, the written study plans with their digests,
machine-checked proofs, and the per-fold and per-episode scores behind every table and figure. The data of the
two placement families are attached to the release `data-v1` (see `DATA.md`). The paper's four-page
supplementary material summarizes the studies; the complete record, with every registered reading, the plan
table with digests and the execution record of every study, is `extended_report.pdf` (sections E1 to E10).

The paper asks when a robot policy whose weights are frozen at deployment generalizes zero-shot from a task
description, and states two conditions that limit it:

1. **Where the description enters.** Conditioning layers that are linear in the representation and feed a linear
   output map give each task one linear readout; a task receives more only if the description acts on a nonlinear
   readout.
2. **What the description encodes.** Training fixes a head's response only to the part of a new description
   that lies in the span of the training descriptions. That share can be computed before training.

## The method and its variant

| Name in the paper | What it is | Name in code and records |
|---|---|---|
| MSAL | The description modulates the first activation of the head's trunk; residual atom library for adaptation | arm `tfilm_nl`, class `TrunkFiLMField` (`code/analysis/tc_arms.py`); library and adaptation in `code/studies/tc_combo.py` (`train_library_t`, `invert_t`) |
| MSAL-h | The variant built first: the description modulates the trunk's output | arm `film_nl` (`code/method/`) |
| Input concatenation | The description is concatenated to the trunk input | arm `tin_nl` |

A rule written down before the LIBERO run of the trunk-modulated head would have kept MSAL-h as the main method.
The paper presents the trunk-modulated head as MSAL nonetheless, with those results known, and reports both
variants. The record of that decision is `plans/DECISION_HEAD_20261121.md`; the studies that repeat on MSAL's head
what had been run only on MSAL-h's were planned and run after it (`plans/PLAN_TRUNK2.md`).

## Layout

| Folder | Contents |
|---|---|
| `code/base/` | Base package: frozen-feature data loading, flow-matching heads, atom libraries |
| `code/method/` | MSAL-h and the other heads compared in the paper, adaptation (descriptor inversion, closed-form code), closed-loop rollouts, perturbed scenes |
| `code/studies/` | One runner per registered study (leave-one-task-out, leave-one-direction-out, LIBERO, projection control, input concatenation, text embedding, the repeats on MSAL's head `tc_lodo_t.py`, `tc_text_t.py`, `tc_bench_t.py`, ...) |
| `code/analysis/` | Scripts that turn the stored records into every number, interval and table of the paper, and the trunk-modulated arms (`tc_arms.py`) |
| `code/libcoord/` | The cluttered placement family in LIBERO: scene generation, scripted demonstrations, head training, rollouts, analysis (`lc_analyze*.py`; seed-level intervals in `lc_seedboot.py`) |
| `code/cl/`, `code/l40/`, `code/text/`, `code/nodes/` | Closed-loop helpers, LIBERO unit lists, text-embedding descriptors, unit lists of the multi-machine runs |
| `code/pen/` | Analysis of the sensitivity of adaptation to the two stage-2 penalty weights (runner: `code/studies/tc_penalty.py`) |
| `code/viz/` | What the description modulates in the stored heads (supplementary Fig. S2, extended report Fig. E3): extraction from the stored weights and summary |
| `plans/` | The written plan of every study (`PLAN_*.md`), the decision record, their SHA-256 digests and the time-stamped digest log; `plans/earlier/` holds the plans of the first twelve studies |
| `results/studies/` | Per-fold and per-episode records and analysis outputs of the later studies |
| `results/earlier/` | Records and aggregated results of the first twelve studies |
| `lean/` | Lean 4 proofs of the algebraic statements (39 theorems) and the axiom audit |
| `docs/` | Project page |

Where the main numbers of the paper come from:

| Paper | Records and analysis output |
|---|---|
| Table I, sampled-chunk columns | `results/studies/heads5/` |
| MSAL's row of Table I and its chained comparisons | `results/studies/switch/switch_numbers.txt` |
| Condition 2 on held directions, text embedding and CPU time with MSAL's head | `results/studies/trunk2/trunk2_ab.txt` (records in `trunk2/a`, `b`, `b2`, `b3`, `bench_t.jsonl`) |
| The same studies with MSAL-h | `results/studies/lodo5_analysis.txt`, `span/`, `proj/`, `text/` |
| Closed loop in the cluttered family | `results/studies/clh/`, `clt/`, `trunk2/cl_trunk2.txt`; seed-level intervals `trunk2/cl_seedlevel.txt` |
| Closed loop on LIBERO | `results/studies/cl/`, `results/earlier/` |
| Sensitivity to the penalty weights (supplementary Section S2, extended report E2) | `results/studies/pen/penalty_analysis.txt` (records in `pen/ws2a`, `ws2b`, `local`) |
| What the description modulates (supplementary Fig. S2, extended report Fig. E3) | `code/viz/mod_viz_ws2.py`, `code/viz/mod_viz_summary.py`, data `code/viz/mod_viz.npz` |

## The plans and their digests

Every plan in `plans/` was hashed before the runs it governs. `plans/PLAN_HASHES.log` lists time, digest and file;
later lines for the same file are dated amendments, and the extended report (Table E17) prints the first 16 digits of
each final digest. The digests were time-stamped on the authors' machines, not in a public registry. In the released
plan files, machine names and directory names in paths were replaced, so their own digests differ from the logged
ones; the logged digests are those of the unedited files, which the authors keep. The log also lists a pilot
(`PLAN_PILOT8`) on a different question, a head shared across task families; it is not part of the paper and its
files are not in this release.

`results/studies/trunk2/trunk2_ab_partialB.txt` is kept on purpose: it is the printout of one part of
`PLAN_TRUNK2` on two of five seeds that was seen before the last units of that part had run, as the plan's third
amendment and the extended report (Table E18) state.

## Checking the proofs

```
cd lean
lake build
```

The toolchain is pinned in `lean/lean-toolchain` (Lean 4.17.0, standard library only). `lean/audit_output.txt`
lists the axioms of every theorem.

## Reproducing numbers from the stored records

The analysis scripts in `code/analysis/` and `code/libcoord/` read the records in `results/` and print the
comparisons with their intervals; their output is stored next to the records as `*_analysis.txt`, `*.txt` and
`*.json`. The scripts keep the directory layout of the study machines, so the paths at the top of each script, or
its command-line arguments, must be pointed at `results/studies/` before they run.

## Retraining

Training needs the frozen SmolVLA features and the demonstration data (see `DATA.md`), PyTorch and, for the
closed-loop tests, LIBERO. The runners in `code/studies/` are the exact scripts that produced the records; they
expect the base package and the method code on the Python path.

## What was changed for release

Machine names, user and account names, network addresses and some directory names were replaced in scripts,
records and plans, and comments were removed from the source files (the Python syntax trees are unchanged, and
the Lean proofs build as before). Orchestration shell scripts, logs and model checkpoints are not included.

## License

Code: MIT (see `LICENSE`). Records, plans, proofs and the released data: CC BY 4.0. The scanned objects of the
cluttered family come from Google Scanned Objects (CC BY 4.0, Google Research).
