from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

CL = Path("/home/ubuntu/work/manuscript6/cl_adapt")
sys.path.insert(0, str(CL))
from smolvla_state_contract import extract_task_independent_spatial_state_feature, raw_policy_observation

BASE = Path("/home/ubuntu/work/cache/hf/hub/models--lerobot--smolvla_base/snapshots/"
            "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6")
BASE_SHA = "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"
EXPERT0 = CL / "expert0_processor"
TASKS = Path("/home/ubuntu/work/manuscript6/repo/sal_v2_identifiable/runs/"
             "20260829_g39_libero40_substrate_v1/libero40_tasks.json")
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
SETTLE_STEPS = 10
CHUNK, ACTION = 16, 7


def time_embedding(t: torch.Tensor, dimension: int = 64) -> torch.Tensor:
    freqs = torch.exp(torch.linspace(0.0, -math.log(10_000.0), dimension // 2, device=t.device))
    angles = t[:, None] * freqs[None, :] * 2.0 * math.pi
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


def atoms_of(L, h, n):
    inner = F.silu(torch.einsum("kch,bh->bkc", L["w1"], h) + L["b1"][None])
    return (torch.einsum("koc,bkc->bko", L["w2"], inner) + L["b2"][None]).reshape(n, -1, CHUNK, ACTION)


class Head(nn.Module):

    def __init__(self, sd: dict, lib: dict | None = None):
        super().__init__()
        self.sd = {k: v.float() for k, v in sd.items()}
        self.lib = None if lib is None else {k: v.float() for k, v in lib.items()}

    def to_(self, device):
        self.sd = {k: v.to(device) for k, v in self.sd.items()}
        if self.lib is not None:
            self.lib = {k: v.to(device) for k, v in self.lib.items()}
        return self

    def inputs(self, s, x, t):
        p = self.sd
        n = s.shape[0]
        s = (s.float() - p["state_mean"]) / p["state_std"]
        sh = F.layer_norm(F.silu(F.linear(s, p["state_encoder.0.weight"], p["state_encoder.0.bias"])), (512,),
                          p["state_encoder.2.weight"], p["state_encoder.2.bias"])
        ah = F.silu(F.linear(x.reshape(n, -1).float(), p["action_encoder.0.weight"], p["action_encoder.0.bias"]))
        return torch.cat([sh, ah, time_embedding(t.float())], dim=-1)

    def hidden(self, s, x, t):
        p = self.sd
        h = F.silu(F.linear(self.inputs(s, x, t), p["trunk.0.weight"], p["trunk.0.bias"]))
        h = F.silu(F.linear(h, p["trunk.2.weight"], p["trunk.2.bias"]))
        return F.layer_norm(h, (512,), p["trunk.4.weight"], p["trunk.4.bias"])

    def velocity(self, s, x, t, d, z=None):
        p = self.sd
        n = s.shape[0]
        h = self.hidden(s, x, t)
        dd = d.expand(n, -1)
        g = h * (1.0 + F.linear(dd, p["film_gamma.weight"])) + F.linear(dd, p["film_beta.weight"])
        out = F.linear(g, p["film_head.weight"], p["film_head.bias"])
        out = out + F.linear(F.silu(F.linear(g, p["film_mlp.inner.weight"], p["film_mlp.inner.bias"])),
                             p["film_mlp.outer.weight"], p["film_mlp.outer.bias"])
        v = out.reshape(n, CHUNK, ACTION)
        if z is not None:
            v = v + torch.einsum("k,bktd->btd", z.float(), atoms_of(self.lib, h, n))
        return v


class TrunkHead(Head):

    def velocity(self, s, x, t, d, z=None):
        p = self.sd
        n = s.shape[0]
        dd = d.expand(n, -1)
        first = F.silu(F.linear(self.inputs(s, x, t), p["trunk.0.weight"], p["trunk.0.bias"]))
        first = first * (1.0 + F.linear(dd, p["trunk_gamma.weight"])) + F.linear(dd, p["trunk_beta.weight"])
        h = F.layer_norm(F.silu(F.linear(first, p["trunk.2.weight"], p["trunk.2.bias"])), (512,),
                         p["trunk.4.weight"], p["trunk.4.bias"])
        out = F.linear(h, p["plain_head.weight"], p["plain_head.bias"])
        out = out + F.linear(F.silu(F.linear(h, p["plain_mlp.inner.weight"], p["plain_mlp.inner.bias"])),
                             p["plain_mlp.outer.weight"], p["plain_mlp.outer.bias"])
        v = out.reshape(n, CHUNK, ACTION)
        if z is not None:
            v = v + torch.einsum("k,bktd->btd", z.float(), atoms_of(self.lib, h, n))
        return v


def build_policy(art: dict, policy: str, task: int, device: str):
    pr = art["probe"]
    if task != art["held"]:
        raise ValueError("held artifact used for another task")
    if art["kind"] == "held_trunk":
        if policy == "tzero":
            head, d, z, v_ref = TrunkHead(art["head"]), art["d_star"], None, pr["zero_shot_v"]
        elif policy.startswith("tmsal_k"):
            k = int(policy.split("_k")[1])
            head, d, z = TrunkHead(art["head"], art["library"]), art["adapt"][k]["d"], art["adapt"][k]["z"]
            v_ref = art["adapt"][k]["probe_v"]
        else:
            raise ValueError(policy)
    elif art["kind"] == "held":
        if policy == "zero":
            head, d, z, v_ref = Head(art["head"]), art["d_star"], None, pr["zero_shot_v"]
        elif policy.startswith("msal_k"):
            k = int(policy.split("_k")[1])
            head, d, z = Head(art["head"], art["library"]), art["adapt"][k]["d"], art["adapt"][k]["z"]
            v_ref = art["adapt"][k]["probe_v"]
        else:
            raise ValueError(policy)
    else:
        raise ValueError(art["kind"])
    with torch.no_grad():
        v = head.velocity(pr["state"], pr["x"], pr["t"], d, z)
    err = float((v - v_ref).abs().max())
    if err > 1e-3:
        raise RuntimeError(f"rebuilt head disagrees with its probe (max abs {err:.2e})")
    return head.to_(device), d.to(device), (None if z is None else z.to(device)), err


def load_extractor(device: str):
    import hashlib
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla import SmolVLAConfig, SmolVLAPolicy
    digest = hashlib.sha256((BASE / "model.safetensors").read_bytes()).hexdigest()
    if digest != BASE_SHA:
        raise RuntimeError("SmolVLA base weights hash mismatch")
    config = SmolVLAConfig.from_pretrained(EXPERT0, local_files_only=True)
    config.device = "cpu"
    config.pretrained_path = BASE
    policy = SmolVLAPolicy.from_pretrained(BASE, config=config, local_files_only=True, strict=True).eval()
    policy.requires_grad_(False)
    vlm = policy.model.vlm_with_expert.get_vlm_model()
    vlm.vision_model.to(device)
    vlm.connector.to(device)
    policy.model.state_proj.to(device)
    config.device = device
    pre, _ = make_pre_post_processors(config, str(EXPERT0), preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre


def feature(policy, pre, obs, instruction, device):
    batch = pre(dict(raw_policy_observation(obs, instruction)))
    with torch.inference_mode(), torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
        f = extract_task_independent_spatial_state_feature(policy, batch)
    return f.float().reshape(1, -1).to(torch.float16).float()


def sample_chunk(head, f, d, z, generator, steps: int):
    x = torch.randn(1, CHUNK, ACTION, generator=generator, device="cpu").to(f.device)
    with torch.no_grad():
        for i in range(steps):
            t = torch.full((1,), 1.0 - i / steps, device=f.device)
            x = x - head.velocity(f, x, t, d, z) / steps
    return x[0].cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--episodes", default="0-49")
    ap.add_argument("--n-exec", type=int, default=16)
    ap.add_argument("--flow-steps", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    a, b = (int(v) for v in args.episodes.split("-"))

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    info = json.loads(TASKS.read_text())["tasks"][str(args.task)]
    suite = benchmark.get_benchmark_dict()[info["suite"]]()
    matches = [i for i in range(suite.n_tasks) if suite.get_task(i).language.lower() == info["instruction"].lower()]
    if len(matches) != 1:
        raise RuntimeError(f"task {args.task}: {len(matches)} LIBERO matches for {info['instruction']!r}")
    task = suite.get_task(matches[0])
    init_states = suite.get_task_init_states(matches[0])
    art = torch.load(args.artifact, weights_only=False)
    head, d, z, probe_err = build_policy(art, args.policy, args.task, args.device)
    extractor, pre = load_extractor(args.device)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    done_eps = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            if r["policy"] == args.policy and r["task"] == args.task:
                done_eps.add(r["episode"])
    for ep in range(a, b + 1):
        if ep in done_eps:
            continue
        t0 = time.time()
        env.seed(ep)
        env.reset()
        obs = env.set_init_state(init_states[ep % len(init_states)])
        for _ in range(SETTLE_STEPS):
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
        gen = torch.Generator(device="cpu").manual_seed(1_000_003 * args.task + ep)
        success, steps = False, 0
        while steps < MAX_STEPS[info["suite"]] and not success:
            chunk = sample_chunk(head, feature(extractor, pre, obs, info["instruction"], args.device), d, z, gen,
                                 args.flow_steps)
            for j in range(args.n_exec):
                obs, _, done, _ = env.step(np.clip(chunk[j], -1.0, 1.0).tolist())
                steps += 1
                if done:
                    success = True
                    break
                if steps >= MAX_STEPS[info["suite"]]:
                    break
        rec = {"task": args.task, "suite": info["suite"], "policy": args.policy, "episode": ep, "success": success,
               "steps": steps, "seconds": time.time() - t0, "n_exec": args.n_exec, "flow_steps": args.flow_steps,
               "artifact": args.artifact.name, "probe_max_abs": probe_err}
        with args.out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    env.close()


if __name__ == "__main__":
    main()
