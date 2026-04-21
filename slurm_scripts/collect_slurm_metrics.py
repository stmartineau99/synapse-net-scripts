#!/usr/bin/env python3
import subprocess
import argparse
import json

def parse_sacct(job_id, out_path=None, is_array=False):
    result = subprocess.run([
        "sacct",
        "-j", job_id,
        "--parsable2",
        "--noheader",
        "--units", "G",
        "--format=JobID,JobName,Submit,Elapsed,TotalCPU,AllocCPUS,ReqMem,MaxRSS,State"
    ], capture_output=True, text=True)

    tasks = []

    for line in result.stdout.strip().splitlines():
        fields = line.split("|")
        job_id_fmt, job_name, submit, elapsed, total_cpu, alloc_cpus, req_mem, max_rss, state = fields

        if not any(job_id_fmt.endswith(s) for s in [".batch", ".extern"]):
            current_job_name = job_name
            current_req_mem = req_mem
            current_submit = submit
            continue

        if not job_id_fmt.endswith(".batch"):
            continue

        task_id = job_id_fmt.replace(".batch", "")
        # JobID for array tasks is reported as "parentID_arrayIndex"
        array_task_id = task_id.split("_")[1] if "_" in task_id else None

        tasks.append({
            "task_id": task_id,
            "array_task_id": array_task_id,
            "submit": current_submit,
            "elapsed": elapsed,
            "total_cpu": total_cpu,
            "alloc_cpus": alloc_cpus,
            "req_mem": current_req_mem,
            "max_rss": max_rss,
            "state": state,
            })

    output = {
        "job_id": job_id,
        "job_name": current_job_name,
        "submit": current_submit,
        "is_array": is_array,
        "tasks": tasks}

    if is_array and tasks:
        max_max_rss = max(tasks, key=lambda t: float(t["max_rss"].replace("G", "")))
        max_elapsed = max(tasks, key=lambda t: t["elapsed"])
        
        # compute utilization
        req_mem_g = float(tasks[0]["req_mem"].replace("G", ""))
        avg_rss_g = sum(float(t["max_rss"].replace("G", "")) for t in tasks) / len(tasks)
        avg_rss_util = avg_rss_g / req_mem_g * 100

        failed_tasks = [t for t in tasks if t["state"] != "COMPLETED"]
        output["summary"] = {
            "n_tasks": len(tasks),
            "alloc_cpus": tasks[0]["alloc_cpus"],
            "req_mem": tasks[0]["req_mem"],
            "max_rss": max_max_rss["max_rss"],
            "avg_rss": f"{avg_rss_g:.2f}G",
            "avg_rss_utilization": f"{avg_rss_util:.1f}%",
            "max_elapsed": max_elapsed["elapsed"],
            "all_completed": len(failed_tasks) == 0,
            "failed": [t["array_task_id"] or t["task_id"] for t in failed_tasks]
        }
 
    if out_path is None:
        out_path = f"./slurm-{job_id}_metrics.json"

    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)

    return output 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse sacct output for a SLURM job and save to JSON.")
    parser.add_argument("job_id", help="SLURM job ID.")
    parser.add_argument("--out_path", default=None, help="Output JSON file path.")
    parser.add_argument("--is_array", action="store_true", help="Treat as a job array rather than single job.")
    args = parser.parse_args()
    parse_sacct(args.job_id, args.out_path, args.is_array)