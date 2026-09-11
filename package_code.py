"""Package the repository as a single flat folder inside a zip archive."""
import os, zipfile, shutil

FILES = [
 "README.md", "LICENSE", "requirements.txt",
 "config.py", "optics.py", "kinetics.py", "breath_cohort.py", "make_data.py",
 "dataset.py", "model.py", "baselines.py", "losses.py", "train.py",
 "evaluate.py", "stats.py", "experiments.py", "external_uci.py", "run_all.py",
 "figures.py", "make_figures.py", "tables.py", "summarise.py",
 "omml.py", "docx_build.py", "cite.py", "equations.py", "refs.py",
 "manuscript_a.py", "manuscript_b.py", "manuscript_c.py", "manuscript_d.py",
 "manuscript_e.py", "build_paper.py", "package_code.py",
]
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = "/mnt/user-data/outputs/kindra-plasmonic-breath-code.zip"
FOLDER = "kindra-plasmonic-breath"


def main():
    missing = [f for f in FILES if not os.path.exists(os.path.join(ROOT, f))]
    if missing:
        raise SystemExit("missing: " + ", ".join(missing))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            z.write(os.path.join(ROOT, f), f"{FOLDER}/{f}")
    size = os.path.getsize(OUT) / 1024.0
    print(f"{OUT}  ({len(FILES)} files, {size:.0f} kB)")


if __name__ == "__main__":
    main()
