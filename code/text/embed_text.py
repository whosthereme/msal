import json
import sys

sys.path.insert(0, "/home/ubuntu/work/manuscript6/cl_adapt")
import numpy as np
import torch
from transformers import AutoTokenizer

from cl_rollout import load_extractor

src, dst = sys.argv[1], sys.argv[2]
ins = {int(k): v for k, v in json.load(open(src)).items()}
policy, pre = load_extractor("cpu")
name = [s for s in pre.steps if type(s).__name__ == "TokenizerProcessorStep"][0].tokenizer_name
tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
vwe = policy.model.vlm_with_expert
tm = vwe.get_vlm_model().text_model.float()
ids, last, static, ntok = [], [], [], []
for tid in sorted(ins):
  enc = tok([ins[tid] + "\n"], return_tensors="pt")
  with torch.no_grad():
    e = vwe.embed_language_tokens(enc["input_ids"]).float()
    out = tm(inputs_embeds=e, attention_mask=enc["attention_mask"], output_hidden_states=True)
  ids.append(tid)
  last.append(out.hidden_states[-1][0].mean(0).numpy())
  static.append(e[0].mean(0).numpy())
  ntok.append(int(enc["input_ids"].shape[1]))
last, static = np.stack(last).astype(np.float32), np.stack(static).astype(np.float32)
np.savez(dst, task_ids=np.array(ids), last=last, static=static, tokens=np.array(ntok), tokenizer=name,
  layers=len(tm.layers))
for nm, m in (("last", last), ("static", static)):
  u = m / np.linalg.norm(m, axis=1, keepdims=True)
  c = u @ u.T
  off = c[~np.eye(len(ids), dtype=bool)]
  print(nm, m.shape, "norm %.2f..%.2f" % (np.linalg.norm(m, axis=1).min(), np.linalg.norm(m, axis=1).max()),
    "cosine between instructions: min %.3f mean %.3f max %.3f" % (off.min(), off.mean(), off.max()))
print("tokens per instruction", min(ntok), max(ntok))
