import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import lc_heads2 as h2
import torch

ARM = "lc_tfilm_xy6"
h2.PROTOCOL = "cl_tfilm_20261121_v1"


class TrunkFiLMXY(h2.tc_arms.TrunkFiLMField):

  def __init__(self, config, source_task_ids):
    h2.PlainNonlinearReadout.__init__(self, config, source_task_ids)
    matrix = h2.xy6(source_task_ids)
    self.register_buffer("task_features", torch.tensor(matrix, dtype=torch.float32), persistent=True)
    self.trunk_gamma = h2.tc_arms.bias_free(int(matrix.shape[1]), config.hidden_dim)
    self.trunk_beta = h2.tc_arms.bias_free(int(matrix.shape[1]), config.hidden_dim)


h2.DESC[ARM] = h2.xy6
h2.run_nl.arms.DESCRIPTORS[ARM] = h2.xy6
h2.run_nl.arms.ARMS[ARM] = dict(family="film", descriptor=ARM, reg=False)
h2.run_nl.BUILDERS[ARM] = lambda c, ids: TrunkFiLMXY(c, ids)
h2.run_nl.CONDITIONED[ARM] = ("trunk_gamma", "trunk_beta")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--fold", type=int, required=True)
  ap.add_argument("--seed", type=int, required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--device", default="cuda")
  ap.add_argument("--smoke-steps", type=int, default=0)
  args = ap.parse_args()
  e2 = h2.e2
  assert e2.TRAIN_STEPS == 12_800
  if args.smoke_steps:
    e2.TRAIN_STEPS = args.smoke_steps
    args.out = args.out.with_name(args.out.name + "_SMOKE")
  args.out.mkdir(parents=True, exist_ok=True)
  if (args.out / ("%s_f%d_s%d.pt" % (ARM, args.fold, args.seed))).exists():
    return
  data = h2.lc_heads.load_data(h2.lc_heads.SHARDS)
  cache = h2.HostFeatureCache(data, args.device)
  (args.out / ("run_info3_f%d_s%d_%d.json" % (args.fold, args.seed, int(time.time())))).write_text(json.dumps(dict(
    protocol=h2.PROTOCOL, fold=args.fold, seed=args.seed, rows=int(len(data.task_ids)), smoke=bool(args.smoke_steps),
    held=h2.held_ids(args.fold), train_steps=int(e2.TRAIN_STEPS)), indent=1))
  h2.run(ARM, args.fold, args.seed, data, cache, args.device, args.out)


if __name__ == "__main__":
  main()
