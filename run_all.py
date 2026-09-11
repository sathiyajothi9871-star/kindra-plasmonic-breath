"""Run the full experimental programme and write every result file."""
import sys, time, json, os
import numpy as np
import config as C, dataset as D, experiments as X

LOGF = open(os.path.join(C.RESULTS_DIR, "run_log.txt"), "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOGF.write(s + "\n"); LOGF.flush()


def main(which):
    raw = D.load_raw()
    t0 = time.time()
    if "0" in which: X.E0_ceiling(raw, log=log)
    if "1" in which: X.E1_main(raw, log=log)
    if "2" in which: X.E2_ablation(raw, log=log)
    if "3" in which: X.E3_sensitivity(raw, log=log)
    if "4" in which: X.E4_robustness(raw, log=log)
    if "5" in which: X.E5_generalisation(raw, log=log)
    if "8" in which: X.E8_uncertainty(raw, log=log)
    log("TOTAL %.1f min" % ((time.time() - t0) / 60.0))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "12345 8")
