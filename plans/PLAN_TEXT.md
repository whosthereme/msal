# PLAN_TEXT: a pretrained text embedding as the task description (LIBERO, zero-shot)

Written 2026-11-19 after the fourth internal review of the manuscript (a referee would ask for language as it is
actually used, not only a 64-word bag). Hashed on ws2 before any head of this plan is trained. Changes after hashing
are written below as dated amendments before the runs they affect.

## Question
The manuscript's only language description is a bag of the instruction's content words (65 numbers). With a
pretrained sentence embedding the description has far more dimensions (960) than there are training tasks (39), so
every new instruction lies partly outside the span of the training descriptions: the regime in which the second
condition (Proposition 2) should matter most. Does such an embedding help zero-shot prediction on held LIBERO tasks,
and does the span analysis say anything about it?

## Description (computed before this plan was hashed; descriptors only, no head, no score)
- Embedding: for each of the 40 instructions, the text "<instruction>\n" (the newline as the policy's own
  preprocessing adds it) is tokenized with the tokenizer of the frozen SmolVLA checkpoint and run, text only,
  through its 16-layer language model; the embedding is the mean of the last hidden state over the instruction's
  tokens (960 numbers). File `text/text_embeddings.npz` (key `last`), script `text/embed_text.py`; CPU.
- Descriptor of task t in the fold that holds task h: `[1, 3 (u_t - mu) / s]`, with `u` the embedding divided by its
  norm, `mu` the mean of `u` over the fold's 39 training tasks and `s` the root-mean-square norm of `u - mu` over
  those tasks (961 numbers; the held descriptor's norm is about 3.1, like the word descriptor's). Everything is
  computed from the training descriptions and the held instruction. Script `text/text_desc.py`.
- Span share of the held descriptor, computed now from descriptors alone (`text/iota_text.json`): mean 0.80 over the
  40 tasks, minimum 0.48, maximum 0.98; the training descriptors have rank 39. For comparison the word descriptor's
  share has mean 0.85 and equals one for 8 tasks. Cosine similarity between raw embeddings of different
  instructions: 0.76 to 0.995, mean 0.91.

## Arms (200 units: 40 held tasks x seeds 148021-148025; training loop and 12,800 steps unchanged)
- E: MSAL's head (`FiLMNonlinearReadout`, unchanged) with the text descriptor above. Zero-shot only (stage 1).
- E-in-span (evaluation only, as PLAN_PROJ): the same trained head scored on `P_S d*`, the held descriptor with its
  out-of-span part deleted.
- E-mean (evaluation only): the same head scored on the constant-only descriptor `[1, 0, ..., 0]`, the training mean
  (what the head predicts when told nothing about the task but the average).
- References, stored: W = MSAL's head with the word descriptor (PLAN_L40_TRUNK records: flow and sampled chunk);
  N = the same readout without task input (`l40_plain_nl`; flow stored, sampled chunk from PLAN_TIN).
Scores: flow MSE on validation rows (eight paired noise draws) and sampled-chunk error (ten Euler steps), test rows
as second check. ws2 GPU 1 through `gpu_run`.

## Readings (fixed now)
Per-seed mean of paired log ratios over tasks, mean over seeds, 95% t interval over the five seeds; a reading is
called only if the interval excludes zero. Sampled chunk is primary where both arms have it, flow otherwise.
- T1: E vs N. Does a pretrained text embedding help zero-shot prediction on a held LIBERO task?
- T2: E vs W. Embedding against the bag of words.
- T3: E-in-span vs E. Proposition 2 says the out-of-span part passes through weights that training did not
  determine; prediction: deleting it does not raise error (lower, or not distinguishable).
- T4: E vs E-mean. Does the head use the held description at all beyond the average task?
- T5: Spearman correlation over the 40 tasks between the span share and the per-task mean log ratio of E against N
  (bootstrap over tasks, 20,000 resamples); "supported" if the interval lies below zero.
Descriptive: T1 to T4 split at the median share; per-suite means; the three tasks with the largest and smallest share.

## Consequences fixed now
The manuscript reports T1 to T5 whatever they show. If T1 is not "lower", the paper states that on LIBERO neither
a bag of words nor a pretrained sentence embedding gave a zero-shot gain, and keeps its statement that the language
result is a null. If T3 reads "higher" (deleting the out-of-span part hurts), the paper reports that the head's
response outside the span, although not fixed by training, was useful here, and says that this counts against
reading the second condition as a rule for removing out-of-span parts.

## Cost
200 zero-shot units of about 1.5 to 2 minutes each: roughly 2 hours of ws2 GPU 1 with three lanes.
