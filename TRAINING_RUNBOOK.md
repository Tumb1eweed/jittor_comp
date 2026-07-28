# ShapeNet 10K Gaussian Training Runbook

This document records the CUDA/Jittor environment, ShapeNet training setup, and
the mesh-based validation evaluation used for this project.

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

1. Loads pre-sampled ShapeNet point clouds from `/home/dataset_train_pgd_points_50k`.
2. Falls back to OBJ mesh loading only when `--precomputed_points_dir` is omitted.
3. Uses `50000` clean points per mesh.
4. Normalizes each point cloud into a unit sphere during pre-sampling.
5. Adds Gaussian noise in unit-sphere coordinates. The current setting samples
   per-sample noise standard deviation uniformly from `[0.005, 0.020]`.
6. Builds `--patches_per_shape` random `1000`-point local patches per training shape.
7. Trains `PGDModel` with InfoCD loss for one epoch.
8. Runs validation on the validate split and saves an `.npz` checkpoint.

Pre-sample once before training:

```bash
cd /home/PGD
python3.7 tools/prepare_shapenet_points.py \
  --dataset_root /home/dataset_train \
  --datalist_dir /home/PGD/datalist \
  --sample_points 50000 \
  --output_dir /home/dataset_train_pgd_points_50k
```

Current command:

```bash
cd /home/PGD
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

`--loss infocd` is the default. `--loss chamfer` is kept only as an explicit
fallback/debug option. The InfoCD implementation is in `models/InfoCD.py` and
uses Jittor tensor operations only.

The current run was started in the background with:

```bash
setsid bash -c 'cd /home/PGD; export CUDA_VISIBLE_DEVICES=3; export cache_name=pgd_cuda; export nvcc_path=/usr/local/cuda/bin/nvcc; export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/root/miniconda3/envs/jittor/lib:/root/miniconda3/envs/jittor/lib/python3.7/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/jittor/targets/x86_64-linux/lib:$LD_LIBRARY_PATH; exec python3.7 tools/train_shapenet_one_epoch.py --use_cuda --dataset_root /home/dataset_train --datalist_dir /home/PGD/datalist --sample_points 50000 --precomputed_points_dir /home/dataset_train_pgd_points_50k --noise_std_min 0.005 --noise_std_max 0.020 --patch_size 1000 --patches_per_shape 4 --batch_size 24 --loss infocd --eval_after_epoch --epochs 1 --log_dir /home/PGD/experiments/shapenet_50k_gaussian_005_020 >> /home/PGD/experiments/shapenet_50k_gaussian_005_020/train.log 2>&1' &
```

MPI multi-GPU training is supported through Jittor's MPI gradient sync. The
script shards training/validation indices by `jt.rank`, while Jittor's optimizer
averages gradients across ranks. Rank 0 writes `history.json` and checkpoints.
The launch follows Jittor's official distributed pattern,
`mpirun -np <N> python ...`; this project wraps the Python command with
`tools/mpi_rank_cuda_wrapper.sh` so each local rank sees one GPU from
`PGD_MPI_DEVICES`.
Use `use_nccl=0` on this machine unless NCCL device all-reduce has been fixed;
the current NCCL path reports `unhandled system error`.
With `--eval_after_epoch`, rank 0 runs `tools/eval_shapenet_mesh_val.py` after
each checkpoint is saved. Per-epoch evaluation outputs are written to
`<log_dir>/eval_epochXX/`, and parsed CD/P2S/final scores are stored in
`history.json`.

```bash
mkdir -p /home/PGD/experiments/shapenet_50k_gaussian_005_020_mpi
setsid bash -c 'cd /home/PGD; export PGD_MPI_DEVICES=0,1,3,4,5,6; export cache_name=pgd_cuda_mpi; export nvcc_path=/usr/local/cuda/bin/nvcc; export use_nccl=0; export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/root/miniconda3/envs/jittor/lib:/root/miniconda3/envs/jittor/lib/python3.7/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/jittor/targets/x86_64-linux/lib:$LD_LIBRARY_PATH; exec mpirun --allow-run-as-root --quiet -np 6 tools/mpi_rank_cuda_wrapper.sh /root/miniconda3/envs/jittor/bin/python3.7 tools/train_shapenet_one_epoch.py --use_cuda --dataset_root /home/dataset_train --datalist_dir /home/PGD/datalist --sample_points 50000 --precomputed_points_dir /home/dataset_train_pgd_points_50k --noise_std_min 0.005 --noise_std_max 0.020 --patch_size 1000 --patches_per_shape 4 --batch_size 24 --loss infocd --eval_after_epoch --epochs 5 --start_epoch 4 --init_weights /home/PGD/experiments/shapenet_10k_gaussian_005_020_one_epoch/pgd-shapenet-epoch03-loss5.37268828.npz --log_dir /home/PGD/experiments/shapenet_50k_gaussian_005_020_mpi >> /home/PGD/experiments/shapenet_50k_gaussian_005_020_mpi/train.log 2>&1' &
```

Runtime artifacts are intentionally ignored by git:

```text
/home/PGD/experiments/shapenet_50k_gaussian_005_020/train.log
/home/PGD/experiments/shapenet_50k_gaussian_005_020/history.json
/home/PGD/experiments/shapenet_50k_gaussian_005_020/pgd-shapenet-epoch*.npz
```

Progress can be checked with:

```bash
tail -f /home/PGD/experiments/shapenet_50k_gaussian_005_020/train.log
```

## Continued training

`tools/train_shapenet_one_epoch.py` supports continuing from a saved `.npz`
checkpoint:

```bash
python3.7 tools/train_shapenet_one_epoch.py \
  --use_cuda \
  --dataset_root /home/dataset_train \
  --datalist_dir /home/PGD/datalist \
  --sample_points 50000 \
  --noise_std_min 0.005 \
  --noise_std_max 0.020 \
  --patch_size 1000 \
  --patches_per_shape 4 \
  --batch_size 24 \
  --loss infocd \
  --eval_after_epoch \
  --epochs 5 \
  --start_epoch 1 \
  --init_weights /home/PGD/experiments/shapenet_50k_gaussian_005_020/pgd-shapenet-epoch00-loss6.36981843.npz \
  --log_dir /home/PGD/experiments/shapenet_50k_gaussian_005_020
```

The interrupted continuation produced checkpoints through epoch 3:

```text
epoch00 train_loss=6.36981843 val_loss=5.87907867
epoch01 train_loss=5.76067139 val_loss=5.34784975
epoch02 train_loss=5.46872081 val_loss=5.16740589
epoch03 train_loss=5.37268828 val_loss=5.28675337
```

### Decoder-only continuation used for CD analysis

The following CUDA/Jittor environment and command were verified on this
machine. It resumes from the epoch-17 PGD checkpoint, freezes the encoder and
trains only decoder blocks/codebooks plus the displacement head for 400 steps.
The training run uses Gaussian noise only, with the same per-sample standard
deviation range as mesh validation.

```bash
cd /home/PGD
conda activate jittor
export CUDA_VISIBLE_DEVICES=3
export cache_name=pgd_cuda
export nvcc_path=/usr/local/cuda/bin/nvcc
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/root/miniconda3/envs/jittor/lib:/root/miniconda3/envs/jittor/lib/python3.7/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/jittor/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

python3.7 tools/train_shapenet_one_epoch.py \
  --use_cuda \
  --dataset_root /home/dataset_train \
  --datalist_dir /home/PGD/datalist \
  --sample_points 50000 \
  --precomputed_points_dir /home/dataset_train_pgd_points_50k \
  --noise_std_min 0.005 \
  --noise_std_max 0.020 \
  --noise_types gaussian \
  --patch_size 1000 \
  --patches_per_shape 4 \
  --batch_size 2 \
  --loss infocd \
  --pgd_composite_loss \
  --pgd_train_decoder_head_only \
  --loss_corr_weight 0.4 \
  --loss_relative_weight 0.7 \
  --loss_uniform_weight 0.05 \
  --lr 5e-6 \
  --train_steps_per_epoch 400 \
  --val_steps_per_epoch 5 \
  --freeze_batchnorm_stats \
  --init_weights /home/PGD/experiments/pgd_epoch17_decoder_distribution_lr8e6_300steps/pgd-shapenet-epoch00-loss0.22946938.npz \
  --epochs 1 \
  --log_dir /home/PGD/experiments/pgd_decoder_distribution_continue_rel07_uniform005_lr5e6_400steps
```

The verified checkpoint is
`experiments/pgd_decoder_distribution_continue_rel07_uniform005_lr5e6_400steps/pgd-shapenet-epoch00-loss0.32580106.npz`.
For a fair fixed-noise 20-shape comparison, use the epoch-17 reference
evaluation as `--reference_eval_root` and run:

```bash
python3.7 tools/eval_shapenet_mesh_val.py \
  --use_cuda \
  --weights /home/PGD/experiments/pgd_decoder_distribution_continue_rel07_uniform005_lr5e6_400steps/pgd-shapenet-epoch00-loss0.32580106.npz \
  --dataset_root /home/dataset_train \
  --val_list /home/PGD/experiments/val_subsets/validate_2syn10each.txt \
  --reference_eval_root /home/PGD/experiments/shapenet_50k_mixed_noise_005_020_mpi_nccl_8ep_eval_b16/eval_epoch17 \
  --sample_points 50000 \
  --patch_size 1500 \
  --seed_k 7 \
  --seed_k_alpha 10 \
  --patch_batch_size 4 \
  --patch_fusion select \
  --niters 1 \
  --workers 8 \
  --run_evaluate \
  --output_root /home/PGD/experiments/scorecheck_2syn20_decoder_continue_rel07_uniform005_patch1500_seedk7
```

This fixed-noise check produced `CD=67.49`, `P2S=91.52`, and final score
`79.51` on 20 shapes. The full 100-shape mesh evaluation should be run
separately without `--reference_eval_root`; its generated clean/noisy clouds
must not be mixed with the fixed-noise A/B comparison.

### Current CD optimization sweep

The current follow-up keeps the encoder frozen, increases the continuation to
2000 steps, and slightly increases the correlation/relative-loss weights while
reducing the uniformity term. It uses the pre-sampled 50k-point clouds and the
same Gaussian noise range as the fixed validation protocol:

```bash
cd /home/PGD
conda activate jittor
export CUDA_VISIBLE_DEVICES=5
export cache_name=pgd_cuda
export nvcc_path=/usr/local/cuda/bin/nvcc
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/root/miniconda3/envs/jittor/lib:/root/miniconda3/envs/jittor/lib/python3.7/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/jittor/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

python3.7 tools/train_shapenet_one_epoch.py \
  --use_cuda \
  --dataset_root /home/dataset_train \
  --datalist_dir /home/PGD/datalist \
  --sample_points 50000 \
  --precomputed_points_dir /home/dataset_train_pgd_points_50k \
  --noise_std_min 0.005 \
  --noise_std_max 0.020 \
  --noise_types gaussian \
  --patch_size 1000 \
  --patches_per_shape 4 \
  --batch_size 2 \
  --loss infocd \
  --pgd_composite_loss \
  --pgd_train_decoder_head_only \
  --loss_corr_weight 0.5 \
  --loss_relative_weight 1.0 \
  --loss_uniform_weight 0.02 \
  --pgd_loss_disp_weight 0.0003 \
  --pgd_loss_low_noise_disp_weight 0.0003 \
  --lr 5e-6 \
  --grad_clip_norm 0.2 \
  --freeze_batchnorm_stats \
  --train_steps_per_epoch 2000 \
  --val_steps_per_epoch 5 \
  --init_weights /home/PGD/experiments/pgd_decoder_distribution_continue_rel07_uniform005_lr5e6_400steps/pgd-shapenet-epoch00-loss0.32580106.npz \
  --epochs 1 \
  --log_dir /home/PGD/experiments/pgd_decoder_cd_rel10_uniform002_2000steps
```

For a fixed-noise A/B check of this checkpoint, use the same reference root,
20-shape list, and `patch_size=1000`:

```bash
python3.7 tools/eval_shapenet_mesh_val.py \
  --use_cuda \
  --weights /home/PGD/experiments/pgd_decoder_cd_rel10_uniform002_2000steps/pgd-shapenet-epoch00-loss<TRAIN_LOSS>.npz \
  --dataset_root /home/dataset_train \
  --val_list /home/PGD/experiments/val_subsets/validate_2syn10each.txt \
  --reference_eval_root /home/PGD/experiments/shapenet_50k_mixed_noise_005_020_mpi_nccl_8ep_eval_b16/eval_epoch17 \
  --sample_points 50000 \
  --patch_size 1000 \
  --seed_k 5 \
  --seed_k_alpha 10 \
  --patch_batch_size 16 \
  --patch_fusion select \
  --niters 1 \
  --workers 8 \
  --run_evaluate \
  --output_root /home/PGD/experiments/scorecheck_2syn20_decoder_cd_rel10_uniform002_patch1000
```

Replace `<TRAIN_LOSS>` with the loss value in the generated checkpoint name.
Do not compare this fixed-noise score with a full 100-shape score until the
same checkpoint has been evaluated under one complete protocol.

The completed `patch_size=1000`, `select` A/B run on the current 400-step
checkpoint produced `CD=66.97`, `P2S=91.05`, and final score `79.01` on the
same 20 fixed-noise shapes. This is below the `patch_size=1500` baseline
(`79.51`), so the smaller patch is not currently preferred.

### Verified score-80 submission baseline

The decoder continuation checkpoint above reaches a fixed-protocol score above
80 when the submitted inference script performs one global two-stage residual
refinement. The model weights are preserved at
`/home/PGD/experiments/pgd_submission_score80_twostage05.npz`; this is a copy
of `pgd-shapenet-epoch00-loss0.91962404.npz`, so no validation samples are
encoded in the weights.

The verified fixed 20-shape command was:

```bash
cd /home/PGD
conda activate jittor
export cache_name=pgd_cuda
export nvcc_path=/usr/local/cuda/bin/nvcc
export PGD_MPI_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/root/miniconda3/envs/jittor/lib:/root/miniconda3/envs/jittor/lib/python3.7/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/jittor/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

python3.7 tools/run_sharded_mesh_eval.py \
  --output_root /home/PGD/experiments/scorecheck_2syn20_decoder2000_twostage05 \
  --dataset_root /home/dataset_train \
  --starter_root /home/starter_code \
  --num_shards 8 \
  --devices 0,1,2,3,4,5,6,7 \
  --workers 8 \
  --use_cuda \
  --weights /home/PGD/experiments/pgd_submission_score80_twostage05.npz \
  --datalist_dir /home/PGD/datalist \
  --val_list /home/PGD/experiments/val_subsets/validate_2syn10each.txt \
  --reference_eval_root /home/PGD/experiments/shapenet_50k_mixed_noise_005_020_mpi_nccl_8ep_eval_b16/eval_epoch17 \
  --sample_points 50000 \
  --patch_size 1500 \
  --seed_k 7 \
  --seed_k_alpha 10 \
  --patch_batch_size 4 \
  --patch_fusion select \
  --pgd_two_stage \
  --pgd_second_stage_scale 0.5 \
  --niters 1
```

The resulting scorer output was `CD=68.12`, `P2S=92.48`, final score
`80.30/100` on 20 shapes. This is a fixed-noise screening result; the complete
100-shape submission protocol must still be run before treating it as the
final leaderboard score.

The complete 100-shape mesh protocol was subsequently run with the same
weights and inference settings. Its authoritative result was `CD=65.84`,
`P2S=91.64`, final score `78.74/100` with 100 valid predictions and no
missing or non-finite outputs. The 20-shape result is therefore only a
screening signal, not a generalization claim.

The matching `patch_size=1000`, `weighted` fusion run produced `CD=67.29`,
`P2S=91.08`, and final score `79.19`. It is also below the baseline; weighted
fusion is slower because it performs CPU-side point accumulation for every
patch.

The 2000-step continuation checkpoint
`experiments/pgd_decoder_cd_rel10_uniform002_2000steps/pgd-shapenet-epoch00-loss0.91962404.npz`
was evaluated with both aligned `patch_size=1000` and the historical
`patch_size=1500` setting:

```text
patch=1000/select: CD=67.14, P2S=91.23, final=79.19
patch=1500/select: CD=67.61, P2S=91.67, final=79.64
```

The `patch_size=1500` result is the best result in this sweep, but it remains
below 80 on the fixed 20-shape comparison. A two-iteration probe was started
with `pred_weight=0.5` but was stopped after only one of two shapes completed;
it is not a valid score and should not be used for comparison.

## Mesh Validation Evaluation

Use `tools/eval_shapenet_mesh_val.py` when validation must be generated from
`/home/dataset_train` meshes rather than from pre-existing `.xyz` point clouds.
The evaluator:

1. Reads entries from `/home/PGD/datalist/validate.txt`.
2. Loads each `/home/dataset_train/<entry>/models/model_normalized.obj`.
3. Samples clean points from mesh faces.
4. Normalizes the sampled clean cloud to the unit sphere.
5. Adds Gaussian noise in normalized coordinates with standard deviation sampled
   from `[0.005, 0.020]`.
6. Runs PGD patch-based denoising.
7. Restores clean/noisy/pred points to the original mesh coordinate frame.
8. Calls `/home/starter_code/evaluate.py` to compute CD and P2S scores.

Command used for the latest run:

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

Latest mesh-val result:

```text
CD score:    33.33 / 100
P2S score:   69.84 / 100
Final score: 51.59 / 100
```

Output artifacts:

```text
/home/PGD/experiments/mesh_val_eval_epoch03_10k_noise005_020/evaluate.log
/home/PGD/experiments/mesh_val_eval_epoch03_10k_noise005_020/count_check.json
/home/PGD/experiments/mesh_val_eval_epoch03_10k_noise005_020/{pred,gt,noisy}/
```

## Working Multi-GPU Training Fallback

On this host, the native Jittor MPI/NCCL route is not currently usable:
Open MPI `mpirun` hangs before launching a two-rank job, MPICH hangs during
`MPI_Init`, and Jittor NCCL initialization reports an unhandled system error.
The repository therefore includes a single-host Jittor fallback that starts
one independent Python worker per GPU and synchronizes the flattened
gradients through a localhost TCP rendezvous server.

The model forward/backward and CUDA kernels still execute on the assigned
GPU. The fallback's communication path is CPU-side, so it is intended for
correctness and reproducible multi-GPU training while the native NCCL setup is
being repaired; it is not expected to match NCCL performance.

Minimal Jittor PGD smoke command (two physical GPUs, two training steps):

```bash
cd /home/PGD
export use_cutt=0
export use_parallel_op_compiler=1
export disable_lock=1

python3.7 tools/launch_manual_multigpu.py \
  --devices 0,1 \
  --log_dir /home/PGD/experiments/mpi_smoke/pgd_2gpu_train_no_global_lock \
  -- python3.7 tools/train_shapenet_one_epoch.py \
  --use_cuda \
  --dataset_root /home/dataset_train \
  --datalist_dir /home/PGD/datalist \
  --sample_points 1000 \
  --precomputed_points_dir /home/dataset_train_pgd_points_50k \
  --noise_std_min 0.005 --noise_std_max 0.020 \
  --noise_types gaussian \
  --patch_size 100 --patches_per_shape 1 \
  --batch_size 2 --loss infocd --lr 5e-4 \
  --grad_clip_norm 0.2 --freeze_batchnorm_stats \
  --pgd_two_stage --pgd_second_stage_scale 0.5 \
  --pgd_use_refine_gate --pgd_refine_gate_scale 0.25 \
  --epochs 1 --train_steps_per_epoch 2 --val_steps_per_epoch 0 \
  --step_metrics_every 1 --seed 2032 \
  --log_dir /home/PGD/experiments/mpi_smoke/pgd_2gpu_train_no_global_lock
```

The launcher sets `CUDA_VISIBLE_DEVICES` separately for each rank, disables
Jittor's shared global compile lock because each rank has its own
`cache_name`, and appends the manual process-group arguments. `--batch_size`
is the per-GPU batch size. Rank 0 writes the checkpoint and history after the
epoch barrier; validation remains disabled in this smoke command and must be
run separately on the complete validation set.

For fresh CUDA caches, `train_shapenet_one_epoch.py` now places a barrier
after model/optimizer initialization, before the first lazy Jittor operator
build. The manual TCP transport waits up to 7200 seconds by default because
first-use compilation and epoch-end mesh validation can differ substantially
by GPU; override with `manual_dist_timeout=<seconds>` when needed.
`launch_manual_multigpu.py` defaults to
`use_parallel_op_compiler=0`; this avoids CPU/compiler contention when several
independent ranks build operators simultaneously. Override it only for a
known-warm cache. The communication fallback is therefore functional across
all available local GPUs, although its CPU-side gradient exchange is slower
than NCCL.

## Current Main Training Flow

This is the default competition-training recipe. It is a **fresh random
initialization** run: do not pass `--init_weights`. It preserves the
best-validated PGD architecture and effective InfoCD objective while using a
larger learning rate for optimization from scratch.

Training uses only `datalist/train.txt`. The validation split is read only by
the mesh evaluator after each completed epoch; it is never used for gradient
updates.

Key settings:

- eight Jittor CUDA workers (`0,1,2,3,4,5,6,7`) synchronized by the manual
  localhost gradient reducer;
- 50,000 precomputed unit-sphere points per shape, Gaussian noise standard
  deviation sampled uniformly from `[0.005, 0.020]`, four random 1,500-point
  train patches per shape, and per-GPU batch size 8;
- two-stage PGD (`second_stage_scale=0.5`) with refine gate
  (`refine_gate_scale=0.25`) and detached second-stage backbone;
- plain Jittor InfoCD objective. Keep `--pgd_composite_loss` disabled. The
  normal-correction flags remain for checkpoint/configuration compatibility,
  but do not enter the effective loss while composite loss is disabled;
- 20 epochs x 400 synchronized train steps. Adam uses a 400-step linear
  warmup from `5e-6` to `1e-4`, then cosine decay to `5e-7` at global step
  8,000;
- after every epoch, save an NPZ checkpoint and run the complete 100-shape
  mesh validation over eight GPU shards, then score CD/P2S with the starter
  evaluator.

Run from `/home/PGD`:

```bash
export manual_dist_timeout=7200
export use_parallel_op_compiler=0
export disable_lock=1
export DISABLE_MULTIPROCESSING=1
export nvcc_path=/usr/local/cuda/bin/nvcc
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/root/miniconda3/envs/jittor/lib:/root/miniconda3/envs/jittor/lib/python3.7/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/jittor/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}

python3.7 tools/launch_manual_multigpu.py \
  --devices 0,1,2,3,4,5,6,7 \
  --log_dir /home/PGD/experiments/pgd_best7911_warmupcosine_fresh_20ep \
  -- python3.7 tools/train_shapenet_one_epoch.py \
  --use_cuda \
  --dataset_root /home/dataset_train \
  --datalist_dir /home/PGD/datalist \
  --data_name models/model_normalized.obj \
  --sample_points 50000 \
  --precomputed_points_dir /home/dataset_train_pgd_points_50k \
  --noise_std_min 0.005 --noise_std_max 0.020 --noise_types gaussian \
  --patch_size 1500 --patches_per_shape 4 --batch_size 8 \
  --epochs 20 --start_epoch 0 --train_steps_per_epoch 400 \
  --lr 0.0001 --lr_schedule warmup_cosine \
  --lr_warmup_steps 400 --lr_warmup_start 0.000005 --lr_min 0.0000005 \
  --grad_clip_norm 0.2 --freeze_batchnorm_stats \
  --model pgd --loss infocd --category_embed_dim 16 --noise_embed_dim 16 \
  --pgd_two_stage --pgd_second_stage_scale 0.5 \
  --pgd_use_refine_gate --pgd_refine_gate_scale 0.25 \
  --pgd_train_detach_second_stage_backbone \
  --pgd_use_normal_corr_loss --normal_k 16 \
  --normal_corr_normal_weight 2.0 --normal_corr_tangent_weight 1.0 \
  --loss_corr_weight 1.0 --loss_relative_weight 0.5 \
  --loss_infocd_weight 0.15 --loss_uniform_weight 0.1 \
  --loss_stage_weight 0.2 --loss_noise_weight 0.05 --corr_huber_delta 0.01 \
  --eval_after_epoch --eval_sample_points 50000 --eval_seed 2026 \
  --eval_seed_k 7 --eval_seed_k_alpha 10.0 --eval_patch_batch_size 128 \
  --eval_patch_fusion select --eval_pgd_gate_noise_source known --eval_niters 1 \
  --eval_workers 8 --eval_num_shards 8 --eval_devices 0,1,2,3,4,5,6,7 \
  --eval_starter_root /home/starter_code \
  --log_dir /home/PGD/experiments/pgd_best7911_warmupcosine_fresh_20ep \
  --step_metrics_every 10 --seed 2025
```

`train_steps.jsonl` records train-only loss and actual step learning rate every
10 steps. `history.json` records the epoch-average loss, learning-rate range,
checkpoint path, and the complete mesh-evaluation result. The expected
per-epoch artifacts are:

```text
pgd-shapenet-epochXX-loss*.npz
eval_epochXX/{pred,gt,noisy}/
eval_epochXX/evaluate.log
history.json
train_steps.jsonl
```

The verified 8-GPU compact training protocol is recorded under
`experiments/pgd_compact_20ep_8gpu_val_each_epoch/`. It uses the required
50k precomputed points, Gaussian noise std `[0.005, 0.020]`, four train
patches per shape, `infocd`, two-stage PGD, refine gate, 400 train steps per
epoch, and complete mesh validation after every epoch. Use the same command
as the experiment's `args.json`, with:

```bash
export use_cutt=0
export use_parallel_op_compiler=0
export disable_lock=1
```

The verified smoke output is:

```text
experiments/mpi_smoke/pgd_2gpu_train_no_global_lock/
  pgd-shapenet-epoch00-loss1.75357395.npz
  history.json
  train_steps.jsonl
```

For a pure communication check, use
`tools/manual_dist_smoke.py` with the same launcher. Its rank-0 and rank-1
final model weights matched after three synchronized updates.
