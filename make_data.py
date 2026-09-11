"""Generate and cache the plasmonic breath cohort."""
import os, time, pickle, numpy as np
import config as C, breath_cohort as B

CACHE = os.path.join(C.CACHE_DIR, "cohort.npz")
DEVCACHE = os.path.join(C.CACHE_DIR, "devices.pkl")


def main(seed=0, force=False):
    if os.path.exists(CACHE) and not force:
        print("cohort already cached at", CACHE)
        return
    t0 = time.time()
    print("generating cohort ...")
    d = B.build_cohort(seed=seed, verbose=True)
    devs = d.pop("devices")
    np.savez_compressed(CACHE, **d)
    meta = [dict(id=v.id, s_bulk=v.s_bulk, s_surf=v.s_surf, l_d=v.l_d,
                 therm=v.therm_dir, age=v.age_dir, spec=v.spec_basis,
                 logK=v.logK, k_a=v.k_a, k_d=v.k_d, n_exp=v.n_exp,
                 drift_basis=v.drift_basis(), lambda0=v.array["lambda0"])
            for v in devs]
    with open(DEVCACHE, "wb") as f:
        pickle.dump(meta, f)
    print("X", d["X"].shape, "in %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)
