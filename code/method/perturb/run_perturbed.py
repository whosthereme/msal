import argparse
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import perturb_lib as pl


def main():
    if "--" not in sys.argv:
        raise SystemExit("usage: run_perturbed.py --table T --level L --script S -- <script args>")
    cut = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--script", type=Path, required=True)
    args = ap.parse_args(sys.argv[1:cut])
    rest = sys.argv[cut + 1:]
    task = int(rest[rest.index("--task") + 1])
    if args.level not in pl.LEVELS:
        raise SystemExit("level must be one of the perturbed levels")
    pl.install(args.table, args.level, task)
    sys.argv = [str(args.script)] + rest
    runpy.run_path(str(args.script), run_name="__main__")


if __name__ == "__main__":
    main()
