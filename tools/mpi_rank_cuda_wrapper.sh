#!/bin/sh
set -eu

devices="${PGD_MPI_DEVICES:-0,1,3,4,5,6}"
rank="${OMPI_COMM_WORLD_LOCAL_RANK:-${PMI_RANK:-0}}"
device="$(printf '%s\n' "$devices" | awk -v r="$rank" -F, '{ print $((r % NF) + 1) }')"
if [ "${use_nccl:-0}" = "1" ]; then
    export CUDA_VISIBLE_DEVICES="$devices"
else
    export CUDA_VISIBLE_DEVICES="$device"
fi
export PATH="/usr/local/cuda/bin:${PATH:-}"
export nvcc_path="${nvcc_path:-/usr/local/cuda/bin/nvcc}"
export cache_name="${cache_name:-pgd_cuda_mpi}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/root/miniconda3/envs/jittor/lib:/root/miniconda3/envs/jittor/lib/python3.7/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/jittor/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

exec "$@"
