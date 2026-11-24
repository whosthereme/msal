import json
import os
import sys
from pathlib import Path


def _main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import task_mod

    mode = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    descriptor = task_mod.install(os.environ["LEROBOT_TASK_LIST"])
    code_path = os.environ.get("TASK_MOD_CODE")
    if code_path:
        import numpy as np
        import torch
        task_mod.set_override(torch.from_numpy(np.load(code_path)).float())
    offset = int(os.environ.get("EVAL_EPISODE_OFFSET", "0"))
    if offset:
        import lerobot.envs.libero as LE
        orig_env_init = LE.LiberoEnv.__init__

        def env_init(self, *a, episode_index: int = 0, **k):
            orig_env_init(self, *a, episode_index=episode_index + offset, **k)
        LE.LiberoEnv.__init__ = env_init
    if mode == "train":
        from lerobot.scripts.lerobot_train import main
    elif mode == "eval":
        from lerobot.scripts.lerobot_eval import main
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    code = main()
    out = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--output_dir=")), None)
    if out and int(os.environ.get("LOCAL_RANK", "0")) == 0 and Path(out).exists():
        (Path(out) / f"task_mod_{mode}.json").write_text(json.dumps(
            {"kind": "modmlp", "vocab": descriptor.vocab, "code_override": code_path}, indent=1))
    return code or 0


if __name__ == "__main__":
    sys.exit(_main())
