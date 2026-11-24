# Pilot: decompose fine-tuned readouts into shared directions (written 2026-11-11, before any run)

Exploratory go/no-go pilot for the follow-up method; nothing here goes into the MSAL paper. One training seed
(148021), the eight held LIBERO tasks of the closed-loop study, GPU 1 only.

## Idea
Fine-tuned readouts work in closed loop (37% at k=16; whole head 59%). If the weight changes that fine-tuning makes
on the source tasks share a few directions, a new task could be adapted by a short code over those directions
instead of by training about one million weights.

## Procedure (per held task h, base head = runs/cl_adapt/held{h}.pt, trained on the other 39 tasks)
1. Source deltas: for each of the 39 source tasks, fine-tune only the readout (film_head, film_mlp; 977,568
   weights) of the base head on that task's first 16 TRAIN demonstrations with the flow loss (Adam 1e-3, 400 steps,
   512 rows per step; the setting selected for k=16 in the readout grid), from the same starting weights. Delta =
   fine-tuned minus base readout.
2. Directions: singular vectors of the 39 deltas (uncentred), each scaled by its singular value / sqrt(39).
3. Code for the held task from its first k TRAIN demonstrations (k = 1, 16), readout = base + sum_i c_i b_i over the
   top r directions (r = 8, 16, 39; r = 16 is primary):
   - flow code: Adam on c, 400 steps, 512 rows per step, learning rate 1e-2 or 1e-1, whichever gives the lower flow
     loss on the demonstrations themselves (4 fixed draws);
   - action code (k = 16, r = 16 only): starting from the flow code, 100 Adam steps (1e-2) on the error of the
     10-step sampled chunk against the demonstrated chunk (4 fixed draws), the objective the closed-loop study
     found to track success.
4. Controls: (a) the same code fit over r = 16 random directions with the per-tensor norms of the source directions;
   (b) the held task's own stored readout fine-tuning (roft{h}.pt, k = 16) projected onto the top r directions;
   (c) my re-implementation of readout fine-tuning on the held task (k = 16) against the stored one (sanity).

## Offline measures (held task's VAL rows, 8 fixed draws, as in the capacity diagnostic)
Sampled-chunk action error and "gap closed" = share of the zero-shot -> seen-head gap removed, pooled over the eight
tasks. Reference values from the capacity diagnostic: MSAL k=16 0.48, action-fitted MSAL code 0.80, whole-head
fine-tuning k=16 1.06. Also: singular-value spectrum, mean pairwise cosine of the source deltas, share of the held
delta's squared norm inside the top r directions (chance level r / 977,568).

## Reading (offline)
- Span: if the projected held delta (control b, r = 39) closes less than half of what the full held delta closes, the
  source tasks' directions do not cover a new task -> no-go for post hoc decomposition (joint training or more
  source tasks would be needed).
- Method: if the r = 16 code at k = 16 closes clearly more of the gap than MSAL (0.48) AND more than the random
  directions, the idea is alive.

## Closed loop (only if warranted)
If the r = 16 flow code at k = 16 closes more than 0.48 of the gap, roll out the r = 16 flow code at k = 1 and 16 and
the action code at k = 16 on the eight held tasks, episodes 0-49, unchanged script and seeds (pairs with Stage B and
the readout follow-up); compare with MSAL, readout fine-tuning and whole-head fine-tuning by cl_adapt.compare.
Otherwise no rollout; the offline result is the pilot's outcome.

## Files
experiments/review/decomp_pilot.py, driver_decomp.sh (remote /home/ubuntu/work/manuscript6/decomp/),
results_remote/decomp/.

## Outcome (added 2026-11-11 after the run; nothing above was changed)
Smoke test first failed on a bug that only affected the 3-source smoke setting (fixed before the full run; the full
run was the first with 39 sources). Full pilot: 8 of 8 tasks, exit 0. Sanity: my readout fine-tuning closes 0.83 of
the gap, the stored one 0.83; reference values match the capacity diagnostic (MSAL 0.48, whole head 1.06).
Gap closed, pooled over the eight held tasks (sampled-chunk action error on VAL):
- held task's own readout delta: full 0.83; projected onto all 39 source directions 0.09 (it keeps 17% of the
  delta's squared norm; same at r = 8 and 16)  -> SPAN FAILS (registered no-go criterion).
- code over source directions: flow, k=16: r=8 -0.02, r=16 0.09, r=39 0.20; k=1: 0.00, 0.10, 0.20.
  action objective, r=16, k=16: 0.30. Random directions: 0.04. All below MSAL (0.48).
- source deltas: mean pairwise cosine 0.34; one common direction holds 36% of the energy, the rest is flat (top 16
  directions 69%, singular values 37, 12, 11, 10, ... 6.4): no low-rank structure beyond a shared component.
Decision by the registered rule: NO-GO, no closed-loop rollout. Post hoc decomposition of independently fine-tuned
readouts does not give directions that cover a new LIBERO task with 39 source tasks. The trained directions do beat
random ones and the action objective again helps (0.09 -> 0.30), so the idea is not empty, but shared directions
would have to be learned jointly (or from many more tasks). Artifacts for rollouts deleted (disk); results in
results_remote/decomp/.
