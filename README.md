<div align="center">

# Guiding Point Cloud Denoising with Learned Structural Priors

![status](https://img.shields.io/badge/status-active-2ea44f)
![python](https://img.shields.io/badge/python-3.7-blue)
![jittor](https://img.shields.io/badge/Jittor-1.3.11.0-orange)
![cuda](https://img.shields.io/badge/CUDA-enabled-green)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

</div>

This repository is the Jittor implementation of PGD point cloud denoising.
Model code, losses, training, validation, and CUDA execution are expected to run
inside the `jittor` conda environment. Do not install or use PyTorch,
PyTorch-Lightning, PyTorch3D, or PyTorch Geometric for the active training path.

## Repository Structure

```text
PGD/
├─ models/
│  ├─ pgd.py
│  ├─ feature.py
│  ├─ blocks.py
│  ├─ InfoCD.py
│  └─ utils.py
├─ datasets/
│  ├─ pcl.py
│  └─ patch.py
├─ utils/
│  ├─ transforms.py
│  └─ misc.py
├─ pointops/
├─ datalist/
├─ pretrained/
├─ tools/
├─ experiments/          # runtime logs/checkpoints/outputs; do not commit
├─ train.py
├─ test.py
├─ evaluate.py
├─ TRAINING_RUNBOOK.md
└─ README.md
```

## Environment

Use the existing conda environment named `jittor`.

```bash
conda activate jittor
python3.7 -c "import jittor as jt; print(jt.__version__)"
```

The verified local environment is documented in
[`TRAINING_RUNBOOK.md`](TRAINING_RUNBOOK.md):

- Conda env: `jittor`
- Python: `3.7.16`
- Jittor: `1.3.11.0`
- NumPy: `1.21.6`
- tqdm: `4.67.3`
- Additional data dependencies: `scipy`, `scikit-learn`, `trimesh`,
  `point-cloud-utils`

For CUDA runs, pass `--use_cuda` to the training or test entrypoint. A quick CUDA
check is:

```bash
CUDA_VISIBLE_DEVICES=3 \
cache_name=pgd_cuda \
nvcc_path=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/root/miniconda3/envs/jittor/lib:$LD_LIBRARY_PATH \
python3.7 -c 'import jittor as jt; jt.flags.use_cuda=1; print("has_cuda", jt.has_cuda); print(jt.array([1.0]).numpy())'
```

Expected result:

```text
has_cuda 1
[1.]
```

CUDA execution currently uses CUDA `12.8` from `/usr/local/cuda`. Keep
`cache_name=pgd_cuda` separate from the CPU/default Jittor cache, and set
`nvcc_path=/usr/local/cuda/bin/nvcc` so the project does not fall back to CPU.
cuDNN is provided by the `jittor` conda environment; on this machine,
`/usr/local/cuda/include/cudnn*.h` and `/usr/local/cuda/lib64/libcudnn*.so*`
are symlinked to `/root/miniconda3/envs/jittor`.

## Runtime Outputs

Keep generated artifacts under `experiments/`.

Recommended paths:

- Training logs: `experiments/<run_name>/train.log`
- Arguments/history: `experiments/<run_name>/args.json`,
  `experiments/<run_name>/history.json`
- Jittor checkpoints: `experiments/<run_name>/*.npz`

Avoid writing new runtime outputs to `logs/` or committing generated checkpoint
files.

## Training

### ShapeNet one-epoch training

This is the current documented CUDA/Jittor training path. It follows the
starter-code datalist split, uses `/home/dataset_train`, samples `50000` clean
points per mesh, and adds Gaussian noise whose standard deviation is sampled
uniformly from `[0.005, 0.020]`.

For faster training, pre-sample the clean point clouds once and train from the
generated `.npy` files instead of re-reading OBJ meshes every batch:

```bash
cd /home/PGD
conda activate jittor
python3.7 tools/prepare_shapenet_points.py \
  --dataset_root /home/dataset_train \
  --datalist_dir /home/PGD/datalist \
  --sample_points 50000 \
  --output_dir /home/dataset_train_pgd_points_50k
```

```bash
cd /home/PGD
conda activate jittor
export CUDA_VISIBLE_DEVICES=3
export cache_name=pgd_cuda
export nvcc_path=/usr/local/cuda/bin/nvcc
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/root/miniconda3/envs/jittor/lib:$LD_LIBRARY_PATH

python3.7 tools/train_shapenet_one_epoch.py \
  --use_cuda \
  --dataset_root /home/dataset_train \
  --datalist_dir /home/PGD/datalist \
  --sample_points 50000 \
  --precomputed_points_dir /home/dataset_train_pgd_points_50k \
  --noise_std_min 0.005 \
  --noise_std_max 0.020 \
  --patch_size 1000 \
  --patches_per_shape 4 \
  --batch_size 24 \
  --loss infocd \
  --eval_after_epoch \
  --epochs 1 \
  --log_dir /home/PGD/experiments/shapenet_50k_gaussian_005_020
```

Background launch with logging:

```bash
mkdir -p /home/PGD/experiments/shapenet_50k_gaussian_005_020
setsid bash -c 'cd /home/PGD; export CUDA_VISIBLE_DEVICES=3; export cache_name=pgd_cuda; export nvcc_path=/usr/local/cuda/bin/nvcc; export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/root/miniconda3/envs/jittor/lib:$LD_LIBRARY_PATH; exec python3.7 tools/train_shapenet_one_epoch.py --use_cuda --dataset_root /home/dataset_train --datalist_dir /home/PGD/datalist --sample_points 50000 --precomputed_points_dir /home/dataset_train_pgd_points_50k --noise_std_min 0.005 --noise_std_max 0.020 --patch_size 1000 --patches_per_shape 4 --batch_size 24 --loss infocd --eval_after_epoch --epochs 1 --log_dir /home/PGD/experiments/shapenet_50k_gaussian_005_020 >> /home/PGD/experiments/shapenet_50k_gaussian_005_020/train.log 2>&1' &
```

`--loss infocd` uses the Jittor implementation in `models/InfoCD.py`.
`--loss chamfer` is available only as an explicit debug fallback.
Omit `--precomputed_points_dir` only when you intentionally want slower
on-the-fly OBJ sampling.

MPI multi-GPU training is available after installing OpenMPI. On this machine,
use CPU MPI gradient all-reduce (`use_nccl=0`) because the NCCL device path
currently reports an `unhandled system error`. The launch follows Jittor's
official distributed pattern, `mpirun -np <N> python ...`; this project wraps
the Python command with `tools/mpi_rank_cuda_wrapper.sh` so each local rank sees
one GPU from `PGD_MPI_DEVICES`.
`--eval_after_epoch` writes per-epoch validation scores under
`<log_dir>/eval_epochXX/` and records parsed scores in `history.json`.

```bash
mkdir -p /home/PGD/experiments/shapenet_50k_gaussian_005_020_mpi
setsid bash -c 'cd /home/PGD; export PGD_MPI_DEVICES=0,1,3,4,5,6; export cache_name=pgd_cuda_mpi; export nvcc_path=/usr/local/cuda/bin/nvcc; export use_nccl=0; export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/root/miniconda3/envs/jittor/lib:/root/miniconda3/envs/jittor/lib/python3.7/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/jittor/targets/x86_64-linux/lib:$LD_LIBRARY_PATH; exec mpirun --allow-run-as-root --quiet -np 6 tools/mpi_rank_cuda_wrapper.sh /root/miniconda3/envs/jittor/bin/python3.7 tools/train_shapenet_one_epoch.py --use_cuda --dataset_root /home/dataset_train --datalist_dir /home/PGD/datalist --sample_points 50000 --precomputed_points_dir /home/dataset_train_pgd_points_50k --noise_std_min 0.005 --noise_std_max 0.020 --patch_size 1000 --patches_per_shape 4 --batch_size 24 --loss infocd --eval_after_epoch --epochs 5 --start_epoch 4 --init_weights /home/PGD/experiments/shapenet_10k_gaussian_005_020_one_epoch/pgd-shapenet-epoch03-loss5.37268828.npz --log_dir /home/PGD/experiments/shapenet_50k_gaussian_005_020_mpi >> /home/PGD/experiments/shapenet_50k_gaussian_005_020_mpi/train.log 2>&1' &
```

### Generic PUNet-style training

```bash
cd /home/PGD
conda activate jittor
python3.7 train.py \
  --use_cuda \
  --dataset PUNet \
  --dataset_root ./data \
  --resolutions 10000_poisson 30000_poisson 50000_poisson \
  --noise_min 0.005 \
  --noise_max 0.02 \
  --patch_size 1000 \
  --train_batch_size 20 \
  --lr 5e-4 \
  --log_root ./experiments/PGD
```

To initialize from Jittor-readable weights:

```bash
python3.7 train.py \
  --use_cuda \
  --init_from_weights /path/to/pgd-jittor-weights.npz \
  --log_root ./experiments/PGD
```

## Testing / Denoising

`test.py` loads Jittor `.npz` weights and writes denoised point clouds.

```bash
cd /home/PGD
conda activate jittor
python3.7 test.py \
  --use_cuda \
  --weights /path/to/pgd-jittor-weights.npz \
  --input_root ./data/examples \
  --output_root ./experiments/results/PGD \
  --dataset PUNet \
  --resolutions 10000_poisson 50000_poisson \
  --noise_lvls 0.03 \
  --niters 1 \
  --patch_size 1000 \
  --seed_k 5 \
  --seed_k_alpha 10 \
  --patch_batch_size 8
```

## Evaluation

For the ShapeNet validation split used in this project, run mesh-based val
evaluation from `/home/dataset_train` with `tools/eval_shapenet_mesh_val.py`.
This script reads OBJ meshes from `/home/dataset_train`, samples clean
points per validation mesh, normalizes each sampled point cloud to the unit
sphere, adds Gaussian noise with per-sample standard deviation uniformly sampled
from `[0.005, 0.020]`, runs PGD denoising, restores points to the original mesh
coordinate frame, and calls `/home/starter_code/evaluate.py` for CD/P2S scoring.

Example using the latest available continued-training checkpoint:

```bash
cd /home/PGD
export CUDA_VISIBLE_DEVICES=3
export cache_name=pgd_cuda
export nvcc_path=/usr/local/cuda/bin/nvcc
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/root/miniconda3/envs/jittor/lib:/root/miniconda3/envs/jittor/lib/python3.7/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/jittor/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

python3.7 tools/eval_shapenet_mesh_val.py \
  --use_cuda \
  --weights /home/PGD/experiments/shapenet_10k_gaussian_005_020_one_epoch/pgd-shapenet-epoch03-loss5.37268828.npz \
  --dataset_root /home/dataset_train \
  --val_list /home/PGD/datalist/validate.txt \
  --sample_points 10000 \
  --noise_std_min 0.005 \
  --noise_std_max 0.020 \
  --patch_size 1000 \
  --seed_k 5 \
  --seed_k_alpha 10 \
  --patch_batch_size 8 \
  --niters 1 \
  --workers 8 \
  --run_evaluate \
  --output_root /home/PGD/experiments/mesh_val_eval_epoch03_10k_noise005_020
```

The latest recorded run produced `CD=33.33`, `P2S=69.84`, and final score
`51.59` on 100 validation samples. Outputs are under
`experiments/mesh_val_eval_epoch03_10k_noise005_020/`, with `evaluate.log`,
`count_check.json`, and generated `pred/`, `gt/`, and `noisy/` directories.

For generic pre-produced outputs, run `evaluate.py` directly. The evaluator
expects predicted point cloud files, matching ground-truth files, matching noisy
files, and optional matching mesh files.

```bash
python3.7 /home/starter_code/evaluate.py \
  --pred_dir ./experiments/some_eval/pred \
  --gt_dir ./experiments/some_eval/gt \
  --noisy_dir ./experiments/some_eval/noisy \
  --mesh_dir /home/dataset_train \
  --pred_filename denoised.npy \
  --gt_filename clean.npy \
  --noisy_filename noisy.npy
```

## Notes for Development

- Keep model, loss, dataset, training, validation, and CUDA code in Jittor.
- Keep the code runnable with `python3.7` in the `jittor` conda environment.
- Translate PyTorch reference implementations to Jittor APIs before adding them
  to this repository.
- Put generated training artifacts under `experiments/`.
- See [`TRAINING_RUNBOOK.md`](TRAINING_RUNBOOK.md) for the currently verified
  machine-specific CUDA/Jittor setup and ShapeNet training command.

## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@inproceedings{pgd2026,
  title={Guiding Point Cloud Denoising with Learned Structural Priors},
  author={Your Name and Co-authors},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for
details.
