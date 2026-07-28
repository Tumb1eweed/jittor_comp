#!/usr/bin/env python
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_eval_log(log_path):
    result = {}
    if not Path(log_path).exists():
        return result
    text = Path(log_path).read_text(errors="ignore")
    patterns = {
        "cd_score": r"CD\s+得分:\s*([0-9.]+)",
        "p2s_score": r"P2S\s+得分:\s*([0-9.]+)",
        "final_score": r"最终得分.*?:\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            result[key] = float(match.group(1))
    return result


def clean_mpi_env():
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("OMPI_", "PMIX_", "PMI_", "MPI_", "OPAL_")):
            env.pop(key, None)
    env["use_mpi"] = "0"
    return env


def run_starter_evaluate(output_root, dataset_root, starter_root, workers):
    cmd = [
        sys.executable,
        str(Path(starter_root) / "evaluate.py"),
        "--pred_dir", str(Path(output_root) / "pred"),
        "--gt_dir", str(Path(output_root) / "gt"),
        "--noisy_dir", str(Path(output_root) / "noisy"),
        "--mesh_dir", str(dataset_root),
        "--pred_filename", "denoised.npy",
        "--gt_filename", "clean.npy",
        "--noisy_filename", "noisy.npy",
        "--workers", str(workers),
        "--verbose",
    ]
    log_path = Path(output_root) / "evaluate.log"
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=starter_root, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError("starter evaluate failed with code {}; see {}".format(proc.returncode, log_path))
    return parse_eval_log(log_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--dataset_root", default="/home/dataset_train")
    parser.add_argument("--starter_root", default="/home/starter_code")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--devices", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--no_run_evaluate", action="store_true")
    args, eval_args = parser.parse_known_args()

    # ``--run_evaluate`` belongs to this wrapper: evaluation must run once
    # after *all* prediction shards finish.  Forwarding it to each child lets
    # the first completed shard score a partial pred directory and overwrite
    # ``evaluate.log`` with an invalid intermediate result.
    eval_args = [arg for arg in eval_args if arg != "--run_evaluate"]

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    # Parallel candidate screens run several two-shard evaluators at once.
    # A cache keyed only by shard index makes all their shard-00 / shard-01
    # workers compile into the same Jittor directory, which can corrupt a
    # generated source file. Keep reuse within one candidate but isolate
    # independently launched candidates.
    cache_tag = re.sub(
        r"[^A-Za-z0-9_]", "_", "{}_{}".format(output_root.parent.name, output_root.name)
    )

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if not devices:
        devices = [d.strip() for d in os.environ.get("PGD_MPI_DEVICES", "0").split(",") if d.strip()]
    if not devices:
        devices = ["0"]

    procs = []
    env_base = clean_mpi_env()
    for shard in range(args.num_shards):
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = devices[shard % len(devices)]
        # Each independent candidate and shard needs its own cache directory.
        # Reuse it only within that candidate to avoid concurrent compilation
        # into a shared generated-source directory.
        shared_cache = os.environ.get("PGD_SHARED_EVAL_CACHE", "").strip()
        if shared_cache:
            # Inference graphs are identical across weight branches; reuse a
            # completed reference shard cache instead of recompiling kernels.
            env["cache_name"] = "{}_shard{:02d}".format(shared_cache, shard)
        else:
            env["cache_name"] = "pgd_cuda_eval_{}_shard{:02d}".format(cache_tag, shard)
        env["DISABLE_MULTIPROCESSING"] = "1"
        env["disable_lock"] = "1"
        env["use_parallel_op_compiler"] = "0"
        log_path = output_root / "shard{:02d}.log".format(shard)
        cmd = [
            args.python,
            str(ROOT / "tools" / "eval_shapenet_mesh_val.py"),
            "--output_root", str(output_root),
            "--dataset_root", args.dataset_root,
            "--starter_root", args.starter_root,
            "--workers", str(args.workers),
            "--num_shards", str(args.num_shards),
            "--shard_index", str(shard),
        ] + eval_args
        with open(log_path, "w") as f:
            proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
        procs.append((shard, proc, log_path))

    failures = []
    for shard, proc, log_path in procs:
        code = proc.wait()
        if code != 0:
            failures.append({"shard": shard, "code": code, "log": str(log_path)})
    report = {
        "output_root": str(output_root),
        "num_shards": args.num_shards,
        "devices": devices,
        "failures": failures,
    }
    if failures:
        with open(output_root / "sharded_eval_report.json", "w") as f:
            json.dump(report, f, indent=2)
        raise RuntimeError("sharded eval failed: {}".format(failures))
    if not args.no_run_evaluate:
        report["eval"] = run_starter_evaluate(output_root, args.dataset_root, args.starter_root, args.workers)
    with open(output_root / "sharded_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    if "eval" in report:
        print(json.dumps(report["eval"], indent=2))


if __name__ == "__main__":
    main()
