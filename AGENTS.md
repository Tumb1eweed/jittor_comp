# Project Instructions

- This project must use Jittor for model, loss, training, validation, and CUDA execution.
- Do not introduce PyTorch implementations or PyTorch-only dependencies for training code, losses, model components, or data pipelines.
- When porting reference code from PyTorch repositories or papers, translate tensor operations to Jittor APIs and keep the implementation runnable with `python3.7` in the `jittor` conda environment.
- Runtime training artifacts such as logs, checkpoints, and experiment outputs should stay under `experiments/` and should not be committed.
- CUDA runs use CUDA `12.8` from `/usr/local/cuda` with `nvcc_path=/usr/local/cuda/bin/nvcc` and a separate Jittor cache name, normally `cache_name=pgd_cuda`.
- Keep cuDNN headers and libraries visible to CUDA 12.8. On this machine, the `jittor` conda environment provides cuDNN, and symlinks from `/usr/local/cuda/include/cudnn*.h` and `/usr/local/cuda/lib64/libcudnn*.so*` point to `/root/miniconda3/envs/jittor`.
- ShapeNet starter-style training should use `/home/dataset_train` plus `/home/PGD/datalist/{train,validate,test}.txt`. For the current one-epoch run, sample `50000` points and use Gaussian noise with per-sample standard deviation uniformly sampled from `[0.005, 0.020]`.
- For faster ShapeNet training, pre-sample clean `50000`-point unit-sphere point clouds with `tools/prepare_shapenet_points.py` and train with `--precomputed_points_dir /home/dataset_train_pgd_points_50k` instead of sampling OBJ meshes every batch.
- Training should use multiple random local patches per shape when more coverage is needed; the current training script defaults to `--patches_per_shape 4` for train and keeps validation at one patch per shape.
- Mesh-based validation scoring should use `tools/eval_shapenet_mesh_val.py` when the user asks to evaluate against `/home/dataset_train`. This path samples `50000` clean points from each validate OBJ, normalizes to the unit sphere, adds Gaussian noise with std uniformly sampled from `[0.005, 0.020]`, denoises with Jittor PGD, restores original mesh coordinates, and calls `/home/starter_code/evaluate.py` for CD/P2S scoring.
