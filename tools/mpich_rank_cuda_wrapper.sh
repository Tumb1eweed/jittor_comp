#!/bin/sh
set -eu

rank="${PMI_RANK:-0}"
size="${PMI_SIZE:-1}"
local_rank="${MPI_LOCALRANKID:-${rank}}"
devices="${PGD_MPI_DEVICES:-0,1,2,3,4,5,6,7}"
device="$(printf '%s\n' "$devices" | awk -v r="$local_rank" -F, '{ print $((r % NF) + 1) }')"

# Jittor 1.3.11 detects distributed execution through OMPI_* variables even
# when the underlying MPI implementation is MPICH.
export OMPI_COMM_WORLD_RANK="$rank"
export OMPI_COMM_WORLD_SIZE="$size"
export OMPI_COMM_WORLD_LOCAL_RANK="$local_rank"
export CUDA_VISIBLE_DEVICES="$device"
export PATH="/usr/local/cuda/bin:${PATH:-}"
export nvcc_path="${nvcc_path:-/usr/local/cuda/bin/nvcc}"
export mpicc_path="${mpicc_path:-/home/PGD/tools/mpicc_mpich_showme.sh}"
export cache_name="${cache_name:-pgd_cuda_mpich}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/root/miniconda3/envs/jittor/lib:/root/miniconda3/envs/jittor/lib/python3.7/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/jittor/targets/x86_64-linux-gnu/lib:${LD_LIBRARY_PATH:-}"

exec "$@"
