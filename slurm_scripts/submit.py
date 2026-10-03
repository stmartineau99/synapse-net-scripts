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
}


def parse_args():
    parser = argparse.ArgumentParser(description="Submit the jobs from the [jobs] table of a run config.")
    parser.add_argument("config", type=str, help="Path to TOML config file.")
    parser.add_argument("--job", type=str, default=None, help="Submit only this job. By default, submit all jobs.")
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser.parse_args()


def load_profile(name):
    path = PROFILE_DIR / f"{name}.toml"
    if not path.exists():
        valid = ", ".join(sorted(p.stem for p in PROFILE_DIR.glob("*.toml")))
        raise FileNotFoundError(f"Profile not found: {name}. Select from list: {valid}")
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_jobs(config_path, job=None):
    with open(config_path, "rb") as f:
        jobs = tomllib.load(f).get("jobs")
    if not jobs:
        raise KeyError(f"No [jobs] table in {config_path}.")

    job_tables = {k: v for k, v in jobs.items() if isinstance(v, dict)}
    if job is not None and job not in job_tables:
        raise KeyError(f"No job '{job}' in {config_path}. Available: {', '.join(job_tables)}")

    selected = []
    for name in [job] if job else job_tables:
        spec = dict(job_tables[name])
        spec.setdefault("env", jobs.get("env"))
        if not spec["env"]:
            raise KeyError(f"No env set for job '{name}' in {config_path}.")
        selected.append((name, spec))
    return selected


def build_script(config_path, job, spec, profile, out_path):
    slurm = {**profile, **spec.get("slurm", {})}
    script = REPO_DIR / spec["script"]
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")

    command = ["python", str(script), "--config", str(config_path)]
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


def sbatch_command(sh_path, dependency=None):
    dep_flags = [f"--dependency=afterok:{dependency}", "--kill-on-invalid-dep=yes"] if dependency else []
    return ["sbatch", *dep_flags, str(sh_path)]


def submit_job(script_text, sh_path, dependency=None):
    sh_path.write_text(script_text)
    result = subprocess.run(sbatch_command(sh_path, dependency), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed for {sh_path}: {result.stderr.strip()}")
    job_id = result.stdout.split()[-1]
    print(f"Submitted {sh_path.stem} as job {job_id}.")
    return job_id


def main():
    args = parse_args()

    config_path = Path(args.config).resolve()
    jobs = load_jobs(config_path, args.job)

    timestamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    dependency = None
    for job, spec in jobs:
        profile = load_profile(spec["slurm_profile"])
        stem = f"{config_path.stem}_{job}_{timestamp}"
        script_text = build_script(config_path, job, spec, profile, LOG_DIR / f"{stem}.out")

        if args.dry_run:
            print(shlex.join(sbatch_command(LOG_DIR / f"{stem}.sh", dependency)))
            print(script_text)
            dependency = f"<{config_path.stem}_{job}>"
            continue

        LOG_DIR.mkdir(exist_ok=True)
        dependency = submit_job(script_text, LOG_DIR / f"{stem}.sh", dependency)


if __name__ == "__main__":
    main()
