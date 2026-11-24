import json
from pathlib import Path

from libero.libero import benchmark

V = Path("/home/ubuntu/work/manuscript6/vla_codes")
split = json.loads((V / "split.json").read_text())["held"]
out = {}
for h, info in split.items():
    suite = benchmark.get_benchmark_dict()[info["suite"]]()
    m = [i for i in range(suite.n_tasks) if suite.get_task(i).language.lower() == info["instruction"].lower()]
    assert len(m) == 1, (h, m)
    out[h] = {"suite": info["suite"], "task_id": m[0]}
(V / "suite_ids.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
