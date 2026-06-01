# Project Instructions

- This project must use Jittor for model, loss, training, validation, and CUDA execution.
- Do not introduce PyTorch implementations or PyTorch-only dependencies for training code, losses, model components, or data pipelines.
- When porting reference code from PyTorch repositories or papers, translate tensor operations to Jittor APIs and keep the implementation runnable with `python3.7` in the `jittor` conda environment.
- Runtime training artifacts such as logs, checkpoints, and experiment outputs should stay under `experiments/` and should not be committed.
