import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE), str(BASE / "trunkcond"), str(BASE / "repo" / "sal_v2_identifiable"), str(BASE / "repo")]
import boot
import numpy as np
import tc_combo
from msal_ext import combo, hyb

hyb.STORED = BASE / "stored" / "n1n2"
tc_combo.e2.TRAIN_STEPS = 20
data = tc_combo.load_lang_v5("blind", verify_hash=False)
episodes = np.load(tc_combo.LANG_NPZ, mmap_mode="r")["episode_indices"].astype(np.int64)
cache = tc_combo.DeviceCache(data, "cuda:0")
data.state_features = None
ref = combo.run_fold(held=0, seed=148021, data=data, cache=cache, episodes=episodes, device="cuda:0",
  stored=hyb.stored_film_nl(148021))
mine = [json.loads(x) for x in (BASE / "out_adapt_SMOKE" / "folds.jsonl").read_text().splitlines()]
mine = [r for r in mine if r["arm"] == "film_nl" and r["held_task"] == 0][0]
same = [ref["zero_shot"] == mine["zero_shot"]]
for k in ("1", "2", "4", "8", "16"):
  same.append(ref["inversion"][k] == mine["inversion"][k])
  same.append(all(ref["combination"][k][s] == mine["combination"][k][s] for s in ("selection", "robustness")))
print("identical:", same, "->", all(same))
