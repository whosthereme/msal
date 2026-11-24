import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import lc_tasks
import lc_script
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
T = lc_tasks.tasks()
pick = [0, 4, 8 + 0, 8 + 8, 24 + 0, 24 + 4, 24 + 8, 24 + 12, 24 + 6, 24 + 10]
for rim in (0.045, 0.055, 0.065):
  ok, steps = [], []
  for ti in pick:
    task = T[ti]
    env = OffScreenRenderEnv(bddl_file_name=str(lc_tasks.write_bddl(task, HERE / "bddl")), camera_heights=128, camera_widths=128)
    for ep in range(EPISODES):
      env.seed(ep)
      obs = env.reset()
      for _ in range(10):
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
      script = lc_script.Script((task["x"], task["y"]), rim=rim)
      frames, n = [], 0
      while not script.done and n < 500:
        obs, _, _, _ = env.step(script(obs).tolist())
        n += 1
        if ep == 0 and ti == pick[-1] and n % 25 == 0:
          frames.append(obs["agentview_image"][::-1])
      s = lc_script.success(obs, (task["x"], task["y"]))
      ok.append(s)
      steps.append(n)
      if frames:
        Image.fromarray(np.concatenate(frames[:10], axis=1)).save(HERE / ("pilot_strip_rim%03d.png" % int(rim * 1000)))
    env.close()
  print("rim %.3f: success %d/%d, mean steps %.0f" % (rim, sum(ok), len(ok), np.mean(steps)), flush=True)
  per = np.array(ok).reshape(len(pick), EPISODES).mean(axis=1)
  print("   per task", " ".join("%d:%.0f%%" % (t, 100 * p) for t, p in zip(pick, per)), flush=True)
