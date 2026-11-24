from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
STEPS, LR, FRAME_STRIDE, DRAWS, BATCH = 200, 0.05, 2, 2, 16
RENAME = {"observation.images.image": "observation.images.camera1",
          "observation.images.image2": "observation.images.camera2"}
DATA = "/home/ubuntu/work/pilot/data/libero"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--held", type=int, required=True)
    ap.add_argument("--split", type=Path, default=HERE / "split.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--lr", type=float, default=LR)
    args = ap.parse_args()
    t0 = time.time()
    import task_mod
    descriptor = task_mod.install(os.environ["LEROBOT_TASK_LIST"])
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    dev = os.environ.get("FIT_DEVICE", "cuda")
    policy = SmolVLAPolicy.from_pretrained(str(args.ckpt)).to(dev).eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(args.ckpt),
                                      preprocessor_overrides={"device_processor": {"device": dev},
                                                              "rename_observations_processor": {"rename_map": RENAME}})
    split = json.loads(args.split.read_text())["held"][str(args.held)]
    fps = 10.0
    ds = LeRobotDataset("HuggingFaceVLA/libero", root=DATA, episodes=[split["demo_episode"]],
                        delta_timestamps={"action": [i / fps for i in range(policy.config.chunk_size)]})
    idx = list(range(0, len(ds), FRAME_STRIDE))
    idx = idx[:int(os.environ.get("FIT_MAX_FRAMES", len(idx)))]
    batches = []
    for i in range(0, len(idx), BATCH):
        items = [ds[j] for j in idx[i:i + BATCH]]
        batch = {}
        for k in items[0]:
            v = [it[k] for it in items]
            batch[k] = torch.stack(v) if isinstance(v[0], torch.Tensor) else v
        b = pre(batch)
        images, img_masks = policy.prepare_images(b)
        state = policy.prepare_state(b)
        target = policy.prepare_action(b)[..., :7]
        mask = (~b["action_is_pad"]).float() if "action_is_pad" in b else torch.ones(target.shape[:2], device=dev)
        lt, lm = b["observation.language.tokens"], b["observation.language.attention_mask"]
        g = torch.Generator(device="cpu").manual_seed(1_000_003 * args.held + i)
        noise = [torch.randn(target.shape[0], policy.config.chunk_size, policy.config.max_action_dim, generator=g).to(dev)
                 for _ in range(DRAWS)]
        batches.append((images, img_masks, lt, lm, state, target, mask, noise))
    instr = split["instruction"]
    d0, _ = descriptor.encode(instr)
    d = d0.clone().to(dev).requires_grad_(True)
    opt = torch.optim.Adam([d], lr=args.lr)
    total = sum(float(bt[6].sum()) for bt in batches) * DRAWS * 7

    def loss_step(train: bool):
        task_mod.set_override(d if train else d.detach())
        acc = 0.0
        for images, img_masks, lt, lm, state, target, mask, noise in batches:
            for nz in noise:
                with torch.set_grad_enabled(train):
                    x = policy.model.sample_actions(images, img_masks, lt, lm, state, noise=nz)[..., :7]
                    part = (((x - target) ** 2) * mask[..., None]).sum() / total
                if train:
                    part.backward()
                acc += float(part.detach())
        return acc

    traj = []
    with torch.no_grad():
        start = loss_step(False)
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        cur = loss_step(True)
        if step == 0:
            print(f"grad norm on d at step 0: {float(d.grad.norm()) if d.grad is not None else None}", flush=True)
        opt.step()
        if step % 20 == 0 or step == args.steps - 1:
            traj.append([step, cur])
            print(f"step {step} demo action loss {cur:.5f}", flush=True)
    with torch.no_grad():
        end = loss_step(False)
    task_mod.set_override(None)
    np.save(args.out, d.detach().cpu().numpy())
    log = {"held": args.held, "instruction": instr, "demo_episode": split["demo_episode"], "frames": len(idx),
           "loss_decoded_code": start, "loss_fitted_code": end, "trajectory": traj,
           "code_change_norm": float((d.detach().cpu() - d0).norm()), "seconds": time.time() - t0,
           "settings": {"STEPS": args.steps, "LR": args.lr, "FRAME_STRIDE": FRAME_STRIDE, "DRAWS": DRAWS}}
    args.out.with_suffix(".json").write_text(json.dumps(log, indent=1))
    print(json.dumps({k: v for k, v in log.items() if k != "trajectory"}), flush=True)


if __name__ == "__main__":
    main()
