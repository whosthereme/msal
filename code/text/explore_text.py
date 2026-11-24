import sys
sys.path.insert(0, "/home/ubuntu/work/manuscript6/cl_adapt")
import torch
import cl_rollout
from cl_rollout import load_extractor

policy, pre = load_extractor("cpu")
vwe = policy.model.vlm_with_expert
vlm = vwe.get_vlm_model()
print("vlm children:", [n for n, _ in vlm.named_children()])
print("vwe attrs:", [a for a in dir(vwe) if "embed" in a.lower() or "token" in a.lower() or "processor" in a.lower()])
print("policy attrs:", [a for a in dir(policy) if "token" in a.lower() or "lang" in a.lower()])
print("pre:", type(pre), [type(s).__name__ for s in getattr(pre, "steps", [])])
tm = vlm.text_model
print("text model:", type(tm).__name__, "layers", len(getattr(tm, "layers", [])), "hidden", tm.config.hidden_size)
for s in getattr(pre, "steps", []):
  if hasattr(s, "tokenizer"):
    tok = s.tokenizer
    print("tokenizer step:", type(s).__name__, {k: v for k, v in vars(s).items() if isinstance(v, (int, str, bool, float))})
    enc = tok(["pick up the ketchup and place it in the basket\n"], return_tensors="pt", padding=True)
    print("ids", enc["input_ids"].shape, enc["input_ids"][0][:20].tolist())
    with torch.no_grad():
      e = vwe.embed_language_tokens(enc["input_ids"])
      print("static emb", e.shape, float(e.norm(dim=-1).mean()))
      out = tm(inputs_embeds=e, attention_mask=enc["attention_mask"], output_hidden_states=True)
      print("hidden states", len(out.hidden_states), out.hidden_states[-1].shape, out.hidden_states[-1].dtype)
