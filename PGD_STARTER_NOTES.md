# PGD Starter Val Notes

## Scope

- Repo: `/home/chenrui/workspace/PGD`
- Conda env: `pgdpcf`
- Evaluation split: `/home/chenrui/workspace/starter_code/datalist/validate.txt`
- Synthetic noisy rule: normalize clean point cloud, add Gaussian noise with `std=0.015` and deterministic `seed + sample_index`.
- Prediction input: noisy point cloud plus PGD model only. No GT or mesh is used in prediction.
- Evaluator: `/home/chenrui/workspace/starter_code/evaluate.py`

## Environment Work

- Installed in isolated env `pgdpcf`: PyTorch `2.3.1+cu118`, Lightning `2.4.0`, PyG `2.6.1`, `point-cloud-utils`, `scipy`, `pandas`, `tensorboard`.
- Added `cuda-nvcc=11.8` and `cuda-cccl=11.8` to compile PGD `pointops`.
- Built `pointops_cuda` successfully.
- PyTorch3D official wheel was unavailable for this exact Python/Torch/CUDA combination, and source build needed broader CUDA dev packages. For starter inference, added a local minimal `pytorch3d` compatibility module inside PGD implementing the ops used by PGD prediction.
- Added a lightweight `chamfer3d.ChamferDistance` compatibility import. It now returns nearest-neighbor distance/index tensors compatible with PGD's training loss as well as model loading.

## Added Files

- `/home/chenrui/workspace/PGD/tools/eval_starter_val_pgd.py`
- `/home/chenrui/workspace/PGD/pytorch3d/*`
- `/home/chenrui/workspace/PGD/chamfer3d/__init__.py`

## Current Status

- `PGDModel.load_from_checkpoint('pretrained/PGD.ckpt')` works in `pgdpcf`.
- Warm-start fine-tuning from `pretrained/PGD.ckpt` works via `--init_from_checkpoint`.
- A 1-epoch starter-train-only fine-tune produced `/home/chenrui/workspace/PGD/logs/PGD/PGD_StarterShapeNet_2026_05_26__17_18_00/pgd-epoch00-val_loss0.00027170.ckpt`.
- Continuing the same run to epoch05 produced `/home/chenrui/workspace/PGD/logs/PGD/PGD_StarterShapeNet_2026_05_26__17_18_00/pgd-epoch04-val_loss0.00024497.ckpt`.
- Best PGD single/checkpoint route is 70.08 with low-LR epoch19 PGD: `patch_size=1200`, `seed_k=6`, `seed_k_alpha=5`, `niters=2`, plus `pred_weight=0.98` noisy blend. Checkpoint: `/home/chenrui/workspace/PGD/logs/PGD/PGD_StarterShapeNet_2026_05_26__17_18_00/pgd-epoch19-val_loss0.00024044.ckpt`.
- Best overall starter val so far is 70.38 with greedy noisy/pred feature-threshold routing over PGD candidate outputs. Final output: `/home/chenrui/workspace/logs/starter_val_eval_greedy_pgd_ep20_step24`. Prediction remains test-usable because the learned routing rules use taxonomy, noisy input, candidate predictions, and fixed thresholds only; no GT or mesh is used at prediction time. Count check is clean.
- Failed branch: fixed training noise at 0.015 plus train `patch_size=1200` from epoch17 produced only 56.11 on starter eval despite close internal val loss. Do not use that branch as a candidate.

## Full Eval Command

```bash
CUDA_VISIBLE_DEVICES=2 \
LD_LIBRARY_PATH=/home/chenrui/miniconda3/envs/pgdpcf/lib:$LD_LIBRARY_PATH \
PYTHONUNBUFFERED=1 \
/home/chenrui/miniconda3/envs/pgdpcf/bin/python tools/eval_starter_val_pgd.py \
  --output-root /home/chenrui/workspace/logs/pgd_starter_val_pretrained_niters2 \
  --patch-size 1000 --seed-k 6 --seed-k-alpha 10 --niters 2 \
  --run-evaluate --workers 8
```

After completion, check:

```bash
cat /home/chenrui/workspace/logs/pgd_starter_val_pretrained_niters2/count_check.json
tail -40 /home/chenrui/workspace/logs/pgd_starter_val_pretrained_niters2/evaluate.log
```

## Jittor Migration Status

2026-05-27 current migration state:

- PGD Python implementation paths have been converted away from PyTorch/Lightning/PyTorch3D imports. `rg -n "torch|pytorch|lightning|torch_geometric|torch_cluster|pytorch3d|pointops_cuda" -g '*.py' /home/chenrui/workspace/PGD` returns no matches.
- Removed old PyTorch compatibility directories from the active tree: `pytorch3d/`, `chamfer3d/`, and `pointops/setup.py`.
- Converted the epoch19 checkpoint to Jittor-readable weights: `/home/chenrui/workspace/PGD/weights/pgd-epoch19-val_loss0.00024044.npz`.
- Jittor weight load check passed: 319 tensors loaded, 0 missing.
- Jittor training smoke passed with `patch_size=100`, `train_batch_size=1`, `max_epochs=1`, `steps_per_epoch=1`; output checkpoint: `/home/chenrui/workspace/PGD/logs/PGD/jittor_train_smoke_p100_grad/pgd-jittor-epoch00-loss0.00033182.npz`.
- Jittor starter eval smoke passed with one sample and `patch_size=100`; count check clean at `/home/chenrui/workspace/logs/pgd_jittor_eval_smoke_p100/count_check.json`.
- Installed Jittor-env dependencies: `scipy==1.7.3`, `scikit-learn==1.0.2`, `trimesh==4.4.1`, `point-cloud-utils==0.30.4`. No PyTorch package was installed in the Jittor env.
- Jittor CUDA is working through `/home/chenrui/.cache/jittor/cuda115_cudnn_overlay/bin/nvcc` plus the overlay `lib64` path. Use `cache_name=pgd_cuda` for CUDA runs so Jittor CPU and CUDA core caches do not overwrite each other.
- Replaced the CPU scipy KNN/FPS inference hot path with Jittor tensor ops. `pointops` grouping/KNN/interpolation and patch-based denoise now run through Jittor ops and compile CUDA kernels when `jt.flags.use_cuda=1`.
- Full starter val passed under Jittor GPU using epoch19 NPZ weights and original raw PGD parameters: `patch_size=1200`, `seed_k=6`, `seed_k_alpha=5`, `niters=2`, `patch_batch_size=8`.
- Full Jittor output: `/home/chenrui/workspace/logs/pgd_jittor_epoch19_gpuops_euclid_p1200_n2_b8`.
- Count check is clean: 100 predictions, no mismatches, no non-finite outputs, no missing files.
- Starter eval result from `/home/chenrui/workspace/logs/pgd_jittor_epoch19_gpuops_euclid_p1200_n2_b8/evaluate_rerun.log`: CD score 51.39, P2S score 88.66, final score 70.03.
- Current Jittor GPU training smoke passed with `patch_size=1000`, `train_batch_size=1`, `max_epochs=1`, `steps_per_epoch=1`, `cache_name=pgd_cuda`, and epoch19 NPZ initialization. Output checkpoint: `/home/chenrui/workspace/PGD/logs/PGD/jittor_train_gpu_smoke_p1000/pgd-jittor-epoch00-loss0.00034118.npz`.

Known remaining gap:

- Same-epoch full Jittor training has not been run yet after proving eval non-regression. Short Jittor CPU/GPU training smokes have passed.
- `~/.cache/jittor/jtcuda` was moved to `~/.cache/jittor/jtcuda.bak_20260527_015654` while debugging CUDA path conflicts. Current PGD code sets `nvcc_path=""` before importing Jittor, so it defaults to CPU unless CUDA env vars are explicitly set.
