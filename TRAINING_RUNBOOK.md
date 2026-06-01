# ShapeNet 10K Gaussian Training Runbook

This document records the CUDA/Jittor environment and the current one-epoch
ShapeNet training setup.

## Conda environment

- Environment name: `jittor`
- Python: `3.7.16`
- Interpreter: `/root/miniconda3/envs/jittor/bin/python3.7`
- Jittor: `1.3.11.0`
- NumPy: `1.21.6`
- tqdm: `4.67.3`

CUDA/cuDNN on this machine:

- GPU: NVIDIA GeForce RTX 4090, compute capability `8.9`
- NVIDIA driver: `525.116.04`
- CUDA toolkit: `12.8.1-1`
- `nvcc`: `/usr/local/cuda/bin/nvcc`, release `12.8`, `V12.8.93`
- cuDNN runtime: `libcudnn8 8.9.7.29-1+cuda12.2`
- cuDNN development headers: `libcudnn8-dev 8.9.7.29-1+cuda12.2`

The cuDNN packages were installed because Jittor CUDA tests failed without
`cudnn.h`. The verification command is:

```bash
python3.7 -m jittor.test.test_cuda
```

Expected result:

```text
Ran 5 tests
OK (skipped=1)
```

## Dataset

- Dataset root: `/home/dataset_train`
- Mesh path pattern: `<dataset_root>/<split-entry>/models/model_normalized.obj`
- Split files copied from `/home/starter_code/datalist` into `datalist/`.

Current split sizes as parsed by the training script:

- `datalist/train.txt`: `15733`
- `datalist/validate.txt`: `100`
- `datalist/test.txt`: `200`

The split entries are relative ShapeNet object directories, for example:

```text
shapenet/04401088/d7ed512f7a7daf63772afc88105fa679
```

## Training method

Training entrypoint:

```bash
python3.7 tools/train_shapenet_one_epoch.py
```

The script:

1. Loads ShapeNet OBJ meshes from the datalist split.
2. Samples points from triangle faces proportional to face area.
3. Samples `10000` clean points per mesh.
4. Normalizes each point cloud into a unit sphere.
5. Adds fixed `2.5%` Gaussian noise (`noise_std=0.025`).
6. Builds one `1000`-point local patch per mesh sample.
7. Trains `PGDModel` with InfoCD loss for one epoch.
8. Runs validation on the validate split and saves an `.npz` checkpoint.

Current command:

```bash
cd /home/PGD
python3.7 tools/train_shapenet_one_epoch.py \
  --use_cuda \
  --dataset_root /home/dataset_train \
  --datalist_dir /home/PGD/datalist \
  --sample_points 10000 \
  --noise_std 0.025 \
  --patch_size 1000 \
  --batch_size 8 \
  --loss infocd \
  --epochs 1 \
  --log_dir /home/PGD/experiments/shapenet_10k_gaussian_025_one_epoch
```

`--loss infocd` is the default. `--loss chamfer` is kept only as an explicit
fallback/debug option. The InfoCD implementation is in `models/InfoCD.py` and
uses Jittor tensor operations only.

The current run was started in the background with:

```bash
setsid bash -c 'cd /home/PGD; exec python3.7 tools/train_shapenet_one_epoch.py --use_cuda --dataset_root /home/dataset_train --datalist_dir /home/PGD/datalist --sample_points 10000 --noise_std 0.025 --patch_size 1000 --batch_size 8 --epochs 1 --log_dir /home/PGD/experiments/shapenet_10k_gaussian_025_one_epoch >> /home/PGD/experiments/shapenet_10k_gaussian_025_one_epoch/train.log 2>&1' &
```

Runtime artifacts are intentionally ignored by git:

```text
/home/PGD/experiments/shapenet_10k_gaussian_025_one_epoch/train.log
/home/PGD/experiments/shapenet_10k_gaussian_025_one_epoch/history.json
/home/PGD/experiments/shapenet_10k_gaussian_025_one_epoch/pgd-shapenet-epoch00-*.npz
```

Progress can be checked with:

```bash
tail -f /home/PGD/experiments/shapenet_10k_gaussian_025_one_epoch/train.log
```
