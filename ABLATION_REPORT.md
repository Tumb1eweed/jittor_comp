# PGD cleanup and ablation record

The local regression baseline is the complete 100-shape validation result:

- score: `79.11`
- CD: `66.44`
- P2S: `91.78`
- checkpoint: `experiments/pgd_normalcorr_continue400_from7908/pgd-shapenet-epoch00-loss4.53743574.npz`
- absolute checkpoint path: `/home/PGD/experiments/pgd_normalcorr_continue400_from7908/pgd-shapenet-epoch00-loss4.53743574.npz`

This is the fixed baseline weight for future experiments. Its configured loss
was `infocd` with `loss_corr_weight=1.0`,
`loss_relative_weight=0.5`, `loss_infocd_weight=0.15`,
`loss_uniform_weight=0.1`, `loss_stage_weight=0.2`, and
`corr_huber_delta=0.01`. Crucially, it enabled
`pgd_use_normal_corr_loss=true`, using normal/tangent correction weights
`2.0/1.0`; the training data used 50,000 points, Gaussian noise std sampled
from `[0.005, 0.020]`, `patch_size=1500`, four patches per shape, and seed
`2032`. However, the baseline also had `pgd_composite_loss=false`, so the
normal-correction branch was not active in the effective objective: the
actual optimized loss was the plain Jittor InfoCD loss. Runs intended to be
comparable must keep `--pgd_composite_loss` disabled; enabling the normal
flag alone only computes unused training normals and adds overhead.

The platform test result (`79.61`) is recorded separately and is not used to
select local changes because the test ground truth is unavailable.

The current from-scratch training main flow is documented in
[`TRAINING_RUNBOOK.md`](TRAINING_RUNBOOK.md#current-main-training-flow). It
keeps this baseline's architecture and effective loss, but uses Adam
warmup+cosine learning-rate scheduling instead of the baseline's low-LR
continuation setup.

## Kept inference path

The competition inference path is now deliberately single-path:

- one Jittor PGD checkpoint;
- one denoising iteration;
- `patch_size=1500`, `seed_k=7`, `seed_k_alpha=10`;
- select-one-patch fusion;
- two-stage PGD with `second_stage_scale=0.5` and the trained refine gate.

## Ablations already measured

These experiments were not kept as the competition path because they did not
beat the validation baseline:

| Variant | Validation score | CD | P2S |
| --- | ---: | ---: | ---: |
| baseline, two-stage + refine gate | 79.11 | 66.44 | 91.78 |
| separate refiner | 78.97 | 66.36 | 91.58 |
| TTA, 4 rotations | 78.85 | 67.02 | 90.68 |
| second denoising iteration | 78.43 | 65.82 | 91.05 |
| refine-gate `pred_weight=1.1` | 79.24 | 66.93 | 91.55 |

The last row is retained as an experiment record only: its CD/P2S tradeoff is
not a robust improvement over the baseline, so it is not used for submission.

The removed inference branches were ensemble loading, rotation TTA, weighted
patch fusion, noise-gate estimation, and repeated denoising. They are not
part of the best validated path and their removal does not alter that path's
configuration.

## Cleanup verification

After removing the inactive model branches and their obsolete tests:

- the compact model was retrained for a strict train-only 67-step smoke run;
- that short continuation checkpoint scored `61.96` (CD `48.71`, P2S `75.22`)
  and was rejected, since continuing from a converged checkpoint for only a
  few steps damaged the weights;
- the original best checkpoint was then evaluated through the cleaned model on
  all 100 validation shapes and scored `79.12` (CD `66.24`, P2S `91.99`);
- the cleaned test suite passes: `19 passed`.

The retained checkpoint is therefore still
`experiments/pgd_normalcorr_continue400_from7908/pgd-shapenet-epoch00-loss4.53743574.npz`.
