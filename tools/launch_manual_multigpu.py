"""Launch one Jittor worker per local GPU without MPI initialization."""

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--log_dir", default="")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("provide the worker command after --")
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise SystemExit("--devices must contain at least one GPU")
    port = args.port or free_port()
    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    server_log = open(log_dir / "manual_dist_server.log", "w") if log_dir else subprocess.DEVNULL
    server_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "manual_dist.py"),
        "server",
        "--host", args.host,
        "--port", str(port),
        "--world-size", str(len(devices)),
    ]
    server = subprocess.Popen(server_cmd, cwd=str(PROJECT_ROOT), stdout=server_log, stderr=subprocess.STDOUT)
    workers = []
    files = []
    try:
        time.sleep(0.5)
        for rank, device in enumerate(devices):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device
            env["use_mpi"] = "0"
            env["use_nccl"] = "0"
            # Jittor 1.3.11's parallel operator compiler is not safe when
            # several independent worker processes compile new operators at
            # the same time. Keep compilation serial per worker; model
            # execution remains on the assigned CUDA device. The value can be
            # overridden for a known-warm cache via the environment.
            env["use_parallel_op_compiler"] = os.environ.get(
                "use_parallel_op_compiler", "0"
            )
            # Every worker gets its own cache_name.  Jittor's default global
            # lock is otherwise shared by all ranks and can deadlock when one
            # rank is compiling while another is entering a collective.
            env["disable_lock"] = os.environ.get("disable_lock", "1")
            env["cache_name"] = "pgd_cuda_manual_gpu{}".format(device)
            env["nvcc_path"] = "/usr/local/cuda/bin/nvcc"
            env["manual_dist_host"] = args.host
            env["manual_dist_port"] = str(port)
            env["manual_dist_rank"] = str(rank)
            env["manual_dist_world_size"] = str(len(devices))
            worker_cmd = command + [
                "--manual_dist",
                "--manual_dist_host", args.host,
                "--manual_dist_port", str(port),
                "--manual_dist_rank", str(rank),
                "--manual_dist_world_size", str(len(devices)),
            ]
            if log_dir:
                out = open(log_dir / "manual_dist_rank{:02d}.log".format(rank), "w")
            else:
                out = None
            files.append(out)
            workers.append(subprocess.Popen(worker_cmd, cwd=str(PROJECT_ROOT), env=env, stdout=out, stderr=subprocess.STDOUT))
        exit_codes = []
        failed = False
        for worker in workers:
            code = worker.wait()
            exit_codes.append(code)
            failed = failed or code != 0
        if failed:
            raise SystemExit("manual multi-GPU workers exited with {}".format(exit_codes))
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        if server.poll() is None:
            server.terminate()
        for f in files:
            if f:
                f.close()
        if server_log is not subprocess.DEVNULL:
            server_log.close()


if __name__ == "__main__":
    main()
