"""Explicit stages for retraining. Default artifacts go to reproduction/, preserving V1."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["cv", "study", "audit", "full", "package", "all"])
    p.add_argument("--work-dir", type=Path, default=ROOT / "reproduction")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    p.add_argument("--resume", action="store_true", help="Skip complete CV recipes; incomplete recipes still raise")
    args = p.parse_args()
    work = args.work_dir.resolve()
    env = dict(os.environ, MOA_RUNS_DIR=str(work / "runs"), MOA_MODELS_DIR=str(work / "models"),
               MOA_DATA_DIR=str(args.data_dir.resolve()))
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[variable] = "4"

    def run(script, *arguments):
        print(f"Running {script} {' '.join(map(str, arguments))}", flush=True)
        subprocess.run([sys.executable, str(ROOT/script), *map(str, arguments)], cwd=ROOT, env=env, check=True)

    recipes = json.loads((ROOT/"recipes.json").read_bytes())
    tags = list(recipes)
    stages = ["cv", "study", "audit", "full", "package"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "cv":
            for tag, recipe in recipes.items():
                if args.resume and (work/"runs"/tag/"config.json").exists():
                    cfg = json.loads((work/"runs"/tag/"config.json").read_bytes())
                    complete = (cfg["seeds"] == list(range(5)) and cfg["folds"] == 5
                        and (work/"runs"/tag/"oof.npy").exists()
                        and all((work/"models"/tag/f"seed{s}/fold{f}/model.pt").exists()
                                for s in range(5) for f in range(5)))
                    if complete:
                        print(f"Keeping complete recipe {tag}", flush=True)
                        continue
                    raise ValueError(f"Incomplete recipe {tag}; preserve it and choose a fresh work directory")
                cli = ["--tag", tag, "--bf16"]
                for key, value in recipe.items():
                    if isinstance(value, bool):
                        if value:
                            cli += ["--"+key.replace("_", "-")]
                    else:
                        cli += ["--"+key.replace("_", "-"), str(value)]
                run("trainer.py", *cli)
        elif stage == "study":
            run("response_study.py", "--prepare-library", "--tags", *tags, "--name", "selected_cv")
            run("response_study.py", "--tags", *tags, "--ablate-library")
        elif stage in ("audit", "full"):
            run("refit_response.py", stage)
        else:
            run("response_release.py", "build", "--release-dir", work/"release")
            run("verify_project.py", "--release-dir", work/"release", "--reference", "fp32")


if __name__ == "__main__":
    main()
