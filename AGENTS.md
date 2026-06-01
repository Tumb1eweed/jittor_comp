# Project Instructions

- This project must use Jittor for model, loss, training, validation, and CUDA execution.
- Do not introduce PyTorch implementations or PyTorch-only dependencies for training code, losses, model components, or data pipelines.
- When porting reference code from PyTorch repositories or papers, translate tensor operations to Jittor APIs and keep the implementation runnable with `python3.7` in the `jittor` conda environment.
- Runtime training artifacts such as logs, checkpoints, and experiment outputs should stay under `experiments/` and should not be committed.
- CUDA runs use CUDA `12.8` from `/usr/local/cuda` with `nvcc_path=/usr/local/cuda/bin/nvcc` and a separate Jittor cache name, normally `cache_name=pgd_cuda`.
- Keep cuDNN headers and libraries visible to CUDA 12.8. On this machine, the `jittor` conda environment provides cuDNN, and symlinks from `/usr/local/cuda/include/cudnn*.h` and `/usr/local/cuda/lib64/libcudnn*.so*` point to `/root/miniconda3/envs/jittor`.
- ShapeNet starter-style training should use `/home/dataset_train` plus `/home/PGD/datalist/{train,validate,test}.txt`. For the current one-epoch run, sample `10000` points and use Gaussian noise with per-sample standard deviation uniformly sampled from `[0.005, 0.020]`.
