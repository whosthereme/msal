import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE), str(BASE / "trunkcond"), str(BASE / "repo" / "sal_v2_identifiable"), str(BASE / "repo")]
import boot
import tc_run
from msal_ext import hyb

hyb.STORED = BASE / "stored" / "n1n2"
tc_run.IDEAL_OUT = BASE / "stored" / "ideal_out"
tc_run.PROTOCOL = tc_run.PROTOCOL + "_local"

if __name__ == "__main__":
  tc_run.main()
