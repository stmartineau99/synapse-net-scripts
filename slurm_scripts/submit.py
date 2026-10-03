import shlex
import tomllib
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

REPO_DIR = Path(__file__).resolve().parents[1]
PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
LOG_DIR = REPO_DIR / "slurm_logs"

OPTIONAL_SBATCH = {
    "partition": "-p {}",
    "gpu": "-G {}",
    "constraint": "-C {}",
    "qos": "--qos={}",
    "account": "--account={}",
    "exclude": "--exclude={}",
}


def load_profile(name):
    path = PROFILE_DIR / f"{name}.toml"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in PROFILE_DIR.glob("*.toml")))
        raise FileNotFoundError(f"Missing profile: {path}. Available: {available}")
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_job(config_path, job=None):
    with open(config_path, "rb") as f:
        jobs = tomllib.load(f).get("jobs")
    if not jobs:
        raise KeyError(f"No [jobs] table in {config_path}.")

    job_tables = {k: v for k, v in jobs.items() if isinstance(v, dict)}
    if job is None:
        if len(job_tables) != 1:
            raise KeyError(f"{config_path} has several jobs, choose one with --job: {', '.join(job_tables)}")
        job = next(iter(job_tables))
    if job not in job_tables:
        raise KeyError(f"No job '{job}' in {config_path}. Available: {', '.join(job_tables)}")

    spec = dict(job_tables[job])
    spec.setdefault("env", jobs.get("env"))
    if not spec["env"]:
        raise KeyError(f"No env set for job '{job}' in {config_path}.")
    return job, spec


def render_args(args):
    parts = []
    for key, value in args.items():
        if value is False:
            continue
        parts.append(f"--{key}")
        if isinstance(value, list):
            parts += [str(v) for v in value]
        elif value is not True:
            parts.append(str(value))
    return parts


def build_script(config_path, job, spec, profile, out_path):
    slurm = {**profile, **spec.get("slurm", {})}
    script = REPO_DIR / spec["script"]
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")

    command = ["python", str(script), "--config", str(config_path)] + render_args(spec.get("args", {}))
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={config_path.stem}_{job}",
        f"#SBATCH -o {out_path}",
        f"#SBATCH -t {slurm.get('time', '06:00:00')}",
        f"#SBATCH --nodes={slurm.get('nodes', 1)}",
        f"#SBATCH --cpus-per-task={slurm.get('cpus', 4)}",
        f"#SBATCH --mem={slurm.get('mem', '32G')}",
    ]
    lines += [f"#SBATCH {fmt.format(slurm[key])}" for key, fmt in OPTIONAL_SBATCH.items() if slurm.get(key)]
    lines += [
        "",
        "source ~/.bashrc",
        f"micromamba activate {spec['env']}",
        "",
        f"cd {script.parent}",
        "",
        shlex.join(command),
        "",
    ]
    return "\n".join(lines)


def submit_job(script_text, sh_path):
    sh_path.write_text(script_text)
    result = subprocess.run(["sbatch", str(sh_path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed for {sh_path}: {result.stderr.strip()}")
    job_id = result.stdout.split()[-1]
    print(f"Submitted {sh_path.stem} as job {job_id}.")
    return job_id


def parse_args():
    parser = argparse.ArgumentParser(description="Submit a job from the [jobs] table of a run config.")
    parser.add_argument("config", type=str, help="Path to TOML config file.")
    parser.add_argument("--job", type=str, default=None, help="Job table to submit, required if there are multiple per config.")
    parser.add_argument("--profile", type=str, default=None, help="Use this profile instead of slurm_profile.")
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config).resolve()
    job, spec = load_job(config_path, args.job)
    profile = load_profile(args.profile or spec["slurm_profile"])

    stem = f"{config_path.stem}_{job}_{datetime.now():%Y%m%d_%H%M%S}"
    script_text = build_script(config_path, job, spec, profile, LOG_DIR / f"{stem}.out")

    if args.dry_run:
        print(script_text)
        return

    LOG_DIR.mkdir(exist_ok=True)
    submit_job(script_text, LOG_DIR / f"{stem}.sh")


if __name__ == "__main__":
    main()
