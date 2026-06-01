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
starter-code datalist split, loads meshes from `/home/dataset_train`, samples
`10000` points per mesh, and adds Gaussian noise whose standard deviation is
sampled uniformly from `[0.005, 0.020]`.

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
  --sample_points 10000 \
  --noise_std_min 0.005 \
  --noise_std_max 0.020 \
  --patch_size 1000 \
  --batch_size 24 \
  --loss infocd \
  --epochs 1 \
  --log_dir /home/PGD/experiments/shapenet_10k_gaussian_005_020_one_epoch
```

Background launch with logging:

```bash
mkdir -p /home/PGD/experiments/shapenet_10k_gaussian_005_020_one_epoch
setsid bash -c 'cd /home/PGD; export CUDA_VISIBLE_DEVICES=3; export cache_name=pgd_cuda; export nvcc_path=/usr/local/cuda/bin/nvcc; export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/root/miniconda3/envs/jittor/lib:$LD_LIBRARY_PATH; exec python3.7 tools/train_shapenet_one_epoch.py --use_cuda --dataset_root /home/dataset_train --datalist_dir /home/PGD/datalist --sample_points 10000 --noise_std_min 0.005 --noise_std_max 0.020 --patch_size 1000 --batch_size 24 --loss infocd --epochs 1 --log_dir /home/PGD/experiments/shapenet_10k_gaussian_005_020_one_epoch >> /home/PGD/experiments/shapenet_10k_gaussian_005_020_one_epoch/train.log 2>&1' &
```

`--loss infocd` uses the Jittor implementation in `models/InfoCD.py`.
`--loss chamfer` is available only as an explicit debug fallback.

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

Run `evaluate.py` on a produced output directory. The evaluator expects
predicted `.xyz` files, matching ground-truth `.xyz` files, and matching mesh
`.off` files.

```bash
python3.7 evaluate.py \
  --pred_dir ./experiments/results/PGD/PUNet_Ours__50000_poisson_0.03 \
  --gt_dir ./data/gt_xyz \
  --mesh_dir ./data/meshes_off \
  --out ./experiments/results/PGD/PUNet_Ours__50000_poisson_0.03_eval.json
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
