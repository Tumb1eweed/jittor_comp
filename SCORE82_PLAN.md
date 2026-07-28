# Score-82 competition plan and evidence record

This document separates verified measurements from proposed experiments. It is
the working record for improving the competition score before pursuing a
publication-oriented method extension.

## Verified results

### Platform submission (authoritative test result)

The best submitted weight was evaluated by the competition platform as:

| Metric | Value |
| --- | ---: |
| score | `79.61` |
| CD score | `68.07` |
| P2S score | `91.16` |
| mean CD, prediction | `0.000079` |
| mean CD, noisy input | `0.000246` |
| mean P2S, prediction | `0.000053` |
| mean P2S, noisy input | `0.000196` |

The submission uses
`experiments/pgd_normalcorr_continue400_from7908/pgd-shapenet-epoch00-loss4.53743574.npz`
with one Jittor PGD denoising iteration, `patch_size=1500`, `seed_k=7`,
`seed_k_alpha=10`, select patch fusion, two-stage scale `0.5`, and refine-gate
scale `0.25`. The exact submission metadata is retained at
`experiments/test_submission_best7911/result/metadata.json`.

### Local regression baseline

The comparable local result is the complete 100-shape mesh validation score:

| Metric | Value |
| --- | ---: |
| score | `79.11` |
| CD score | `66.44` |
| P2S score | `91.78` |

This is a local validation measurement, not a replacement for the platform
test result. The fixed baseline and its historical ablations are documented in
`ABLATION_REPORT.md`.

### Ten-sample directional-CD diagnosis

Ten deterministic validation visualizations were generated with the submitted
weight and inference configuration under `experiments/visualize_best7911/`.
The mean squared CD directions were:

| Stage | pred to GT | GT to pred | total |
| --- | ---: | ---: | ---: |
| noisy | `0.00011395` | `0.00004982` | `0.00016377` |
| denoised | `0.00002953` | `0.00002705` | `0.00005657` |

The denoiser improves pred-to-GT by `74.1%` and GT-to-pred by `45.7%` on this
small diagnostic set. After denoising, pred-to-GT is slightly larger, but both
directions are material. High P2S alone cannot determine the dominant CD
direction because P2S uses the continuous mesh while CD uses the discrete clean
point sample. See `experiments/visualize_best7911/manifest.json` for the
per-shape values and `before.ply`/`after.ply` for overlays.

## Non-negotiable protocol

- Model training uses only shapes listed in `datalist/train.txt`.
- `datalist/validate.txt` must not be used for training, early stopping,
  hyperparameter selection, or category-vocabulary construction.
- A fixed, category-stratified 10% holdout is split from `train.txt` for model
  selection. The remaining 90% is the only source of train patches.
- The complete 100-shape validation set is run once only for the training-holdout
  winner of a round. It is a final report for that round, not a tuning signal.
- All training, losses, models, and CUDA execution remain Jittor-only.

## Staged score objective

### Candidate decision rule (updated 2026-07-27)

Candidate methods are not rejected merely because an intermediate score is
below `80`.  The authoritative gate is a paired comparison against the best
weight's **validation-set** baseline under the same split, seed, 50k-point
protocol, and scorer.  The submitted test score `79.61` is not this gate; it
must not be mixed with validation results.  A candidate is retained whenever
it gives a reproducible validation improvement in total score or the target
CD component (with pred→GT and GT→pred reported); proxy holdouts are screening
evidence only and cannot override a positive official-validation result.

The audit also found an older inference candidate that must be revisited:
`pred_weight=1.1` reached `79.91` (CD `67.78`, P2S `92.03`) on an existing
20-shape validation subset, versus the 100-shape baseline `79.11`. This is
encouraging but not yet an authoritative improvement because the sample count
differs; it is queued for a complete 100-shape validation recheck.

The immediate target is a reproducible complete-100-shape validation score
strictly above `80.0`, with no material P2S regression.  Score `82.0` remains
the second-stage target after this threshold is established. A 20-shape
screening result is never sufficient evidence for either target.

All three branches below start from the submitted/baseline checkpoint, use
50,000-point precomputed ShapeNet clouds, Gaussian noise sampled uniformly
from `[0.005, 0.020]`, `patch_size=1500`, four train patches per shape, and
600 low-learning-rate train-only fine-tuning steps.

### Eight-GPU execution policy

All eight RTX 4090 GPUs should be kept useful whenever a score-82 round is
running. Use Jittor distributed data-parallel training for one branch when the
branch needs the shortest wall-clock time; use the remaining GPUs for
independent branches only when their checkpoints and output directories are
disjoint. Internal-holdout and final mesh evaluation run as eight independent
prediction shards, one GPU per shard, followed by one score aggregation step.

For the three-branch round, the default allocation is:

| Resource | GPU allocation | Purpose |
| --- | --- | --- |
| branch A | `0,1,2` | train-only fidelity fine-tune |
| branch B | `3,4,5` | train-only coverage fine-tune |
| branch C | `6,7` | train-only robustness fine-tune |

Run the internal-holdout evaluation for a branch only after its training ranks
finish. Reassign all eight GPUs to the winning branch's complete validation
evaluation; do not run competing complete-validation jobs in parallel.

### A. Fidelity branch

Enable PGD composite loss with paired correspondence correction, directional
relative CD terms, and score-aligned relative CD. Keep uniform regularization
weak. The purpose is to reduce pred-to-GT while protecting the existing P2S
strength.

### B. Coverage branch

Start from the same recipe as branch A and add a low-weight local density
consistency term to the PGD composite objective. This requires wiring the
existing `density_consistency_loss` into `pgd_training_loss` (it is currently
available for the ASDN path but not included in the PGD total). The purpose is
to improve GT-to-pred coverage without forcing globally uniform sampling.

### C. Robustness branch

Start from branch B and add a light yaw-consistency penalty plus paired yaw
augmentation. The goal is to test whether geometry-consistent updates improve
the weak CD direction without relying on validation-specific tuning.

For each branch, select only with the fixed training holdout. Reject a branch
when its holdout P2S falls below baseline tolerance or either CD direction
regresses materially. Run the complete validation protocol exactly once for the
holdout winner.

### Round-1 internal-holdout result (2026-07-26)

The category-stratified training-only screen contains 26 shapes.  All entries
below use the same 50k-point meshes, Gaussian draws, seed `8200`, and official
starter scorer.  They are selection evidence only, **not** the 100-shape final
validation result.

| Variant | CD score | P2S score | total score | pred→GT CD | GT→pred CD |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: select fusion | 63.68 | 91.77 | **77.73** | 4.1905e-5 | 3.8494e-5 |
| A: centre-weighted overlap fusion | 63.86 | 91.38 | 77.62 | 4.2410e-5 | 3.7539e-5 |
| A + 200-step overlap-consistency fine-tune, select | 63.68 | 91.77 | **77.73** | 4.1903e-5 | 3.8496e-5 |
| A + 200-step overlap-consistency fine-tune, weighted | 63.86 | 91.38 | 77.62 | 4.2408e-5 | 3.7541e-5 |

Weighted fusion improves coverage (`GT→pred`) but worsens off-surface error
(`pred→GT`) and P2S.  Its total-score regression rejects it as a main-path
change.  The short overlap-consistency run did not produce a measurable
aggregate change, because its raw value (~4e-4 to 6e-4) times the provisional
weight `0.05` was orders of magnitude below the dominant InfoCD contribution.
The implementation remains available for a normalized, contribution-calibrated
follow-up; it is not merged into the submission path.

### Anti-cluster loss sweep (2026-07-26)

A one-sided 8-NN spacing-collapse term was added: it penalizes only predicted
neighbour distances below 85% of the paired clean point's local spacing.  This
is targeted at the visible compact clusters without a global-uniformity prior.
Three 300-step, two-GPU runs (`0.05`, `0.10`, `0.20`) and a matched zero-weight
control were evaluated on the same 26-shape training-only screen.  All four
scored `77.73` (CD `63.68`, P2S `91.77`) at scorer precision.  Their directed
CD totals were respectively `8.039941e-5`, `8.039961e-5`, `8.039966e-5`, and
`8.039958e-5`.

The loss and tests are retained, but the sweep is rejected as a main-path
change: its effect is below control-run variability.  The evidence indicates
that low-LR, short continuation does not materially alter the checkpoint;
the next round must test a more direct inference-time anti-collapse mechanism
or a materially stronger training schedule, always selected on the training
holdout before complete validation.

### Displacement-strength diagnostic (2026-07-26)

The baseline checkpoint was evaluated on the identical 26-shape training
screen with `pred_weight` = `0.85`, `0.90`, `0.95`, and `1.00`.  The total
scores were respectively `75.81`, `76.85`, `77.48`, and `77.73`.  Reducing
the global displacement worsens both CD and P2S; visible clusters are therefore
not explained by a globally excessive denoising step.  Reject global scaling
as a main-path change.  Any subsequent geometric correction must be local and
surface/tangent-aware rather than a global blend toward the noisy cloud.

### Geometry-only postprocessing screen (2026-07-26)

All measurements in this section use the same fixed 26-shape **training**
holdout.  No validation shape was used.  A local PCA plane projection on the
prediction alone did not improve CD: its best tested strength (`0.15`) was
`77.83` (CD `63.72`, P2S `91.94`).  It is therefore not a standalone candidate.

In contrast, a single Jittor tangent-plane repulsion step, with planes fitted
on the predicted cloud and a fixed noisy-cloud kNN graph, consistently reduced
the discrete CD clustering error.  The best standalone setting tested so far
was `77.84` (CD `64.11`, P2S `91.57`) against the select-fusion baseline
`77.73` (CD `63.68`, P2S `91.77`).  Applying that repulsion after the weak
PCA projection yielded `77.96` (CD `64.16`, P2S `91.77`) for steps `0.10--0.125`.

This is the first positive, score-preserving anti-cluster signal, but it is
only `+0.23` on the train holdout and is not enough evidence for a full
validation run.  The postprocessor is retained as a candidate component while
the stronger train-time CD-loss round runs.  It preserves point count exactly
and does not use GT at inference.

#### Geometry-refinement strength check (2026-07-26)

The cached baseline predictions on the same frozen 26-shape training holdout
were reused for a narrow no-GT sweep.  PCA strengths `0.10`, `0.15`, `0.20`
followed by tangent-repulsion steps `0.10`, `0.10`, `0.10` scored `77.93`,
`77.96`, `77.97`; the stronger `0.15` step `0.15` scored `77.96`.  The best
setting (`0.20`, `0.10`) had CD `64.09`, P2S `91.85`, pred→GT `4.16964e-5`,
and GT→pred `3.78220e-5`, compared with raw `77.73`.  This confirms that a
small surface-aware, tangent-only coverage correction is real, but the
`+0.24` gain is too small and too close to prior settings to consume a new
full-validation run.  All sweep outputs were removed; the parameter evidence
and existing inference utilities are retained for combining with a genuinely
stronger learned checkpoint.

### Strong CD-loss continuation round (running)

The prior anti-cluster continuation used weak weights (`0.05--0.20`) for 300
steps and was indistinguishable from the zero-weight control.  The active
round therefore uses 600 steps at `1e-5`, four independent two-GPU Jittor
branches: strong one-sided anti-cluster, density consistency, local surface
distance, and their balanced combination.  It trains exclusively on the fixed
90% training subset and will be selected only on the fixed training holdout.
For each completed checkpoint, the automated screen also compares the fixed
geometry-only candidate (PCA strength `0.15`, then predicted-normal tangent
repulsion step `0.125`) with raw output and records both directional CD
components.  This remains a train-holdout-only comparison.

After that screen completes, one deterministic winner (maximum holdout score,
then name only as a tie-breaker) is selected and sent to the complete
100-shape validation once.  If that winner uses postprocessing, raw full-set
predictions are generated only as its intermediate input and are **not**
separately scored.

### Strong CD-loss continuation final confirmation (2026-07-26, rejected)

The train-holdout winner (the `combined` 600-step continuation followed by
PCA `0.15` and one tangent-repulsion step `0.125`) was run exactly once on the
complete 100-shape validation set. This was a final confirmation only; the
result was not used to choose any setting.

| Metric | Value |
| --- | ---: |
| score | `78.51` |
| CD score | `65.62` |
| P2S score | `91.40` |
| mean CD prediction | `0.00006686` |
| directional pred→GT CD | `3.57029e-5` |
| directional GT→pred CD | `3.11551e-5` |

Although discrete CD is lower, P2S regresses versus the local baseline
(`91.78`) and the combined score is below both that baseline (`79.11`) and
the immediate `80.0` target. The candidate is rejected and is not merged into
the submission path. Its runtime outputs were deleted to keep experiment
storage clean; this record preserves the result and decision.

### Balanced-transport and density-aware CD follow-up (running)

The rejected continuation confirms that local spacing penalties and geometric
postprocessing alone are insufficient.  The current train-only screen therefore
tests two losses that explicitly model the discrete-CD cluster failure:

1. **Balanced local Sinkhorn coverage.**  A 128-point, entropically regularized
   local transport cost gives clean targets and predictions equal mass.  Unlike
   one-sided kNN spacing, it makes a many-to-one cluster costly because it
   cannot carry all target mass.  Four two-GPU branches (zero control and
   weights `0.02`, `0.05`, `0.10`) train only on `train_90.txt`; no validation
   list is read.  Selection will be only on the fixed 26-shape train holdout.
2. **Density-aware Chamfer (DCD) multiplicity.**  The prepared fallback uses
   nearest-neighbour selection counts in both directions, following the DCD
   principle of Wu et al. (NeurIPS 2021), and normalizes its exponential scale
   by clean local spacing.  It is less expensive than Sinkhorn and directly
   penalizes many-to-one matches.  The active implementation uses four reliable
   single-GPU continuations at weights `0.02`, `0.05`, `0.10`, and `0.20`;
   the already-screened 1000-step zero-weight continuation is its control. It
   uses a stronger 1000-step warmup-cosine schedule because the earlier
   600-step `1e-5` continuations were often indistinguishable from their
   controls.

No result from either screen is a validation result.  Any branch that does not
beat its train-holdout control will be deleted immediately.  At most one
holdout winner will receive a complete 100-shape confirmation run.

#### DCD screen: weights `0.05` and `0.10` (2026-07-26, rejected)

Both 1000-step density-aware-Chamfer continuations scored `77.72` on the
frozen 26-shape training holdout, below the `77.73` control (CD `63.68`, P2S
`91.76` in both cases).  Weight `0.05` produced pred→GT `4.19125e-5` and
GT→pred `3.84860e-5`; its tiny pred→GT reduction was outweighed by worse
coverage.  Weight `0.10` was worse in both directions (pred→GT `4.19133e-5`,
GT→pred `3.84872e-5`).  Thus the multiplicity term did not improve this
continuation at useful weights.  Their checkpoints and prediction outputs are
deleted; the lower `0.02` and exploratory `0.20` branches were subsequently
screened under the identical train-only protocol.

#### DCD screen: weights `0.02` and `0.20` (2026-07-26, rejected)

The lower weight scored `77.72` (CD `63.68`, P2S `91.76`), with pred→GT
`4.19130e-5`, GT→pred `3.84859e-5`. Weight `0.20` merely matched the control
at reported precision (score `77.73`, CD `63.69`, P2S `91.77`), with pred→GT
`4.19115e-5`, GT→pred `3.84861e-5`. Its minute directional change is below
score precision and does not support a generalizable improvement. All DCD
continuation checkpoints and train-holdout predictions are therefore deleted.
The Jittor DCD implementation remains as a documented negative result; no
validation shape was used in selection or training.

#### Surface-uniformity continuation grid (running)

To directly test the observed clumping without using validation supervision,
two complementary 1000-step Jittor grids start from the submitted checkpoint
and train only on `train_90.txt`:

1. **Global NN uniformity:** weights `0.05`, `0.10`, `0.20` (the baseline
   used only `0.02`).  This tests whether moderate equalization can reduce
   duplicated/clumped predictions.
2. **Clean-local density consistency:** weights `0.05`, `0.10`, `0.20`, with
   the global weight kept at `0.02`.  It matches each predicted point's local
   nearest-neighbour spacing to the paired clean training surface, preserving
   legitimate density variation that a global term might erase.

The six branches are independent single-GPU runs and use the exact same
frozen 26-shape training holdout, mesh scorer, and directional-CD report as
the DCD screen.  Only a meaningful holdout winner can be retained; all other
runtime checkpoints and predictions will be deleted.

#### Global-uniformity weight `0.05` (2026-07-26, rejected)

The first completed branch scored `77.72` (CD `63.68`, P2S `91.76`) on the
frozen 26-shape train holdout, below the `77.73` control.  Its directional CD
was pred→GT `4.19138e-5` and GT→pred `3.84876e-5`, both slightly worse than
the control (`4.1905e-5`, `3.8494e-5` only at the displayed precision for the
second direction, with the unrounded control total still lower).  Thus a
stronger global nearest-neighbour uniformity prior does not give a
generalizable anti-cluster benefit in this continuation.  Its checkpoint and
runtime predictions are deleted; the two remaining global weights and the
three clean-local density variants continue under the same protocol.

#### Global-uniformity weight `0.10` (2026-07-26, rejected)

Weight `0.10` only matched the control at scorer precision: score `77.73`,
CD `63.68`, P2S `91.77`. Its unrounded directional CD was pred→GT
`4.19105e-5`, GT→pred `3.84892e-5`, versus the control's `4.1905e-5` and
`3.8494e-5`. The tiny coverage movement is offset by worse off-surface error
and is far below a meaningful selection margin. The branch is rejected and
its checkpoint and screen output are deleted.

#### Global-uniformity weight `0.20` (2026-07-26, rejected)

The strongest global weight scored `77.72` (CD `63.68`, P2S `91.76`). Its
directional CD was pred→GT `4.19108e-5`, GT→pred `3.84898e-5`, total
`8.04006e-5`. Like weight `0.10`, it trades a negligible coverage change for
worse pred→GT and does not improve the aggregate score. The entire global-NN
uniformity grid is rejected; all three checkpoints and screen outputs have
been removed.

#### Clean-local density consistency weight `0.05` (2026-07-26, rejected)

The density-preserving alternative also scored `77.72` (CD `63.68`, P2S
`91.77`) on the frozen holdout. Its directional CD was pred→GT `4.19099e-5`,
GT→pred `3.84918e-5`, total `8.04017e-5`; that is not lower than the control
total and the score rounds downward. Matching only the nearest-neighbour
spacing is therefore insufficient to resolve cluster multiplicity. Its
checkpoint and screen output are deleted while weights `0.10` and `0.20`
finish under the same protocol.

#### Clean-local density consistency weights `0.10`, `0.20` (2026-07-26, rejected)

Both remaining weights round to the `77.73` control score. Weight `0.10`
gave CD `63.69`, P2S `91.77`, pred→GT `4.19014e-5`, GT→pred `3.84957e-5`;
weight `0.20` gave CD `63.68`, P2S `91.78`, pred→GT `4.18897e-5`,
GT→pred `3.85058e-5`, total `8.03954e-5`. The latter tiny unrounded total
movement is obtained by trading coverage for off-surface error and is far
below a robust selection margin; neither branch exceeds the control score.
Both are rejected and their checkpoints and runtime outputs are removed. The
separate, same-seed long-horizon control-versus-density experiment remains in
flight to test optimization duration rather than treating this noise-level
change as an improvement.

#### Sinkhorn screen: `ot005` (2026-07-26, rejected)

On the frozen 26-shape training holdout, the `0.05` Sinkhorn branch exactly
matched the control within reported precision: score `77.73`, CD `63.68`, and
P2S `91.77`.  Directional CD was pred→GT `4.19069e-5` and GT→pred
`3.84922e-5`, again indistinguishable from the control.  It therefore has no
evidence of a useful CD improvement and is rejected.  Only the stronger
`ot010` screen remains in flight; no validation result was used in this
decision.

#### Sinkhorn screen: `ot010` (2026-07-26, rejected)

The strongest weight (`0.10`) also scored `77.73` (CD `63.68`, P2S `91.77`)
on exactly the same frozen holdout.  Its directed components, pred→GT
`4.19065e-5` and GT→pred `3.84928e-5`, were again indistinguishable from the
control.  Balanced local transport therefore did not move this continuation
in a useful direction at any screened weight, so all Sinkhorn checkpoints and
their runtime outputs are deleted.  The Jittor implementation and this result
record remain for reproducibility; no complete validation was consumed.

### Surface-aware output uniformity (prepared, not yet selected)

Visible clumps are not necessarily fixed by a global uniformity loss: equal
spacing in 3D can blur a sharp feature or incorrectly change the sampling of a
curved region.  A separate Jittor-only candidate therefore preserves **local
tangent-plane neighbour spacing**.  Clean normals are estimated from each
training patch only; the loss matches the first eight projected neighbour
distances of prediction and clean patch, with an additional one-sided collapse
penalty.  This directly targets clumps while retaining locally legitimate
density variations.  It has no GT, normal, or postprocessing input at
inference.  It will be tested only as a zero-weight-controlled training
ablation on `train_90.txt` and screened once on the fixed training holdout.

#### Tangent-spacing screen (2026-07-26, rejected)

The stronger 1000-step continuation control scored `77.73` on the frozen
26-shape training holdout (CD `63.69`, P2S `91.76`; pred→GT
`4.19155e-5`, GT→pred `3.84808e-5`).  Tangent-spacing weights `0.05` and
`0.10` each scored `77.72`, with mean CD `8.04007e-5` and `8.03982e-5`
respectively versus the control's `8.03963e-5`.  Although the `0.10` branch
very slightly reduced GT→pred, it worsened pred→GT and did not improve the
combined score.  Both are rejected; the runtime checkpoints and predictions
are deleted, while the training-only Jittor loss and selector are retained.

#### Displacement-consensus patch fusion smoke test (2026-07-26, rejected)

A no-GT inference-only seam mitigation was also tested on the frozen 26-shape
training holdout. It averages compatible noisy-to-predicted displacements
using local normal agreement and a displacement-residual gate. This reduced
neither the cluster error nor the aggregate score: the original result was
score `77.73`, CD `63.68` (mean CD `8.040e-5`), while consensus at mix `0.15`
was score `77.15`, CD `63.27` (mean CD `8.122e-5`). Its directed CD was
pred→GT `4.278e-5`, GT→pred `3.844e-5`; the worsening is specifically in
prediction coverage. It is rejected and its runtime output is removed. The
standalone utility remains outside the default inference path for possible
future seam-focused research.

#### Baseline-faithful plain-InfoCD LR continuation (2026-07-26, screening)

The submitted checkpoint's effective objective is plain InfoCD, rather than
the composite objective used in several earlier loss ablations.  We therefore
continued it for 1000 training-only steps on `train_90.txt`, with all other
inference and data settings frozen, and screened once on the fixed 26-shape
training holdout (seed 8200).  This is a real, consistent gain: the prior
control was score `77.73`, CD `63.68`, P2S `91.77`, pred→GT `4.1905e-5`,
GT→pred `3.8494e-5`; the three constant-LR continuations were:

| LR | Score | CD | P2S | pred→GT | GT→pred | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `1e-6` | 77.90 | 63.84 | 91.96 | `4.15888e-5` | `3.83328e-5` | reject (dominated) |
| `2e-6` | 77.96 | 63.89 | 92.02 | `4.14493e-5` | `3.82916e-5` | reject (dominated) |
| `5e-6` | **78.03** | **63.98** | **92.09** | **`4.13464e-5`** | **`3.81583e-5`** | retain; pending full-100 confirmation |

Both directed CD components improve at `5e-6`, so this is not a P2S-only
trade.  Only the `5e-6` checkpoint will be retained for the next decision;
the two dominated branches are removed after this record.  No validation data
was used to train or choose the learning rate.

The higher-LR follow-up confirms that larger steps do not extend this gain.
On the same frozen 26-shape training holdout, `1e-5` reached score `78.03`,
CD `63.97`, P2S `92.09`, pred-to-GT `4.13121e-5`, and GT-to-pred
`3.82289e-5`; `2e-5` reached `77.94`, CD `63.83`, P2S `92.05`,
pred-to-GT `4.13186e-5`, and GT-to-pred `3.85222e-5`.  Although `1e-5`
ties the rounded score, it is CD-dominated by `5e-6` (and worsens the
coverage direction), while `2e-5` is plainly worse. Both runtime artifacts
are removed; no additional full validation is warranted.

#### Plain-InfoCD DCD multiplicity regularisation (2026-07-27, screening)

Four 1000-step plain-InfoCD continuations were trained only on `train_90`,
using a DCD-style multiplicity term to penalize the many-clean-to-one-pred
matches observed in clustered outputs. On the fixed 26-shape train holdout:

| DCD weight | Score | CD | P2S | pred-to-GT | GT-to-pred | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0.01` | **78.05** | **63.99** | **92.11** | **`4.12685e-5`** | `3.81889e-5` | retain for combination only |
| `0.02` | 78.04 | 63.98 | 92.09 | `4.13051e-5` | **`3.81510e-5`** | reject (score-dominated) |
| `0.05` | 77.97 | 63.91 | 92.03 | `4.14132e-5` | `3.82795e-5` | reject |
| `0.10` | 78.03 | 63.96 | 92.10 | `4.13012e-5` | `3.82188e-5` | reject |

The `0.01` branch produces a small score/CD gain over the prior plain-InfoCD
winner (`78.03` / CD `63.98`) and reduces the prediction-to-GT component, but
slightly worsens GT-to-pred. This is too small to spend a new full-validation
run on by itself; retain it only for a train-holdout-selected combination with
a complementary no-GT geometry correction. The other three runtime branches
are removed.

Its one complete 100-shape validation confirmation (seed 8200, same 50k-point
mesh protocol) is score **`78.52`**, CD **`65.35`**, P2S `91.68`, mean CD
`6.74122e-5`, pred→GT `3.53962e-5`, GT→pred `3.20160e-5`.  This is below the
historical full-100 local baseline (`79.11`, from a different run and thus not
a paired comparison), and remains below the `82` target.  It nevertheless
confirms the *direction* of the training-only CD gain under this exact full
protocol; complete validation will not be reused to tune the remaining
candidates.

#### Plain-InfoCD rotation augmentation (2026-07-26, rejected)

The training-only random-Z-rotation continuation scored `77.69`, CD `63.71`,
P2S `91.68`, pred→GT `4.21489e-5`, and GT→pred `3.80944e-5` on the same
frozen screen.  It marginally improves the coverage direction but materially
worsens prediction-to-surface error and total score, so it is rejected and
its runtime artifact is removed.

#### Plain-InfoCD density-jitter augmentation (2026-07-26, rejected)

Training-only density jitter was substantially harmful on the frozen screen.
At jitter ratios `0.05` / `0.10`, scores were `76.91` / `76.82`, CD
`62.76` / `62.68`, P2S `91.06` / `90.97`; directed CD was respectively
`4.19194e-5` / `3.92628e-5` and `4.20644e-5` / `3.92921e-5`
(pred→GT / GT→pred).  Both worsen the clumping-relevant coverage component
and overall score, so both runtime artifacts are removed.  This does not
invalidate the separately controlled long-horizon *loss* experiment: it is a
different intervention.

#### Long-horizon density-loss control (2026-07-27, rejected)

The strictly paired 4000-step composite continuation was trained only on
`train_90`, with the density-aware loss weight either `0` (control) or `0.10`.
The completed density branch scored `77.73`, CD `63.69`, P2S `91.77`, mean CD
`8.03976e-5`, pred-to-GT `4.19044e-5`, and GT-to-pred `3.84932e-5` on the
frozen 26-shape train holdout. Its same-seed zero-weight control finished at
`77.72`, CD `63.69`, P2S `91.75`, mean CD `8.04012e-5`, pred-to-GT
`4.19463e-5`, and GT-to-pred `3.84548e-5`. This <0.01 score difference and
mixed directed-CD change are not a CD improvement and do not justify a full
validation run. Both paired runtime artifacts are removed.

#### Displacement-consensus fusion smoke test (2026-07-26, rejected)

A no-GT patch-seam correction was tested on the same fixed 26-shape training
holdout. It fuses local displacements using noisy-cloud KNN, predicted-normal
agreement, and a displacement-residual gate, directly targeting overlap-patch
seams without changing point count. At mix `0.15` it scored `77.15`, CD
`63.27`, mean CD `8.122e-5`, pred-to-GT `4.278e-5`, and GT-to-pred
`3.844e-5`, versus raw `77.73` / CD `63.68` / mean CD `8.040e-5`.
Both the score and prediction-to-GT direction deteriorate. The method is not
merged into the inference path and is retained only as a documented negative
seam-fusion result.

#### Local Sinkhorn coverage loss (2026-07-27, preliminary screen only)

Four train-only continuations were screened with balanced local OT coverage
weights `0`, `0.02`, `0.05`, and `0.10`. To avoid repeating the very slow 50k
point inference for every candidate, the fixed 26-shape holdout was first
screened with a deterministic 5k subset of the pre-sampled clean clouds; the
same protocol was used for all four branches. Results were effectively tied:
mean CD was `4.138409e-4` (control), `4.138360e-4` (`0.02`), `4.138503e-4`
(`0.05`), and `4.138516e-4` (`0.10`), with pred→GT / GT→pred respectively
`2.011344e-4/2.127065e-4`, `2.011300e-4/2.127058e-4`,
`2.011366e-4/2.127130e-4`, and `2.011364e-4/2.127153e-4`.
The apparent differences are small, but the absolute 5k score is not
comparable to the official 50k score and this screen changes the model input
distribution. Therefore this is **not evidence that Sinkhorn is ineffective**;
it only fails to establish a useful ranking in the 5k proxy. A 50k paired
holdout or full validation run is still required before rejecting it. The
implementation and checkpoints should be retained until that confirmation.

The required complete-100 validation confirmation is now available. With the
same official 50k mesh protocol, Sinkhorn weight `0.02` scored `78.25` (CD
`65.12`, P2S `91.39`), with directional pred→GT `3.582664e-5` and GT→pred
`3.216132e-5`. This is below the best-weight validation baseline `79.11`, so
the Sinkhorn branch is not retained in the main path.

The historical candidates audited in parallel also failed the same gate:
`context125` scored `78.13` (CD `65.05`, P2S `91.21`, pred→GT
`3.601447e-5`, GT→pred `3.205882e-5`), while the full-100 `pred_weight=1.1`
inference variant scored `77.63` (CD `64.44`, P2S `90.82`, pred→GT
`3.620546e-5`, GT→pred `3.294338e-5`). Their earlier 20-shape proxy gains
did not generalize and neither is retained.

## Publication-oriented work starts after score 82

No diffusion model, invertible model, or complex routing architecture is part
of the score-82 round. Once a stable 82+ baseline exists, the candidate
research direction is a bidirectional geometry-aware transport model:

1. Decompose each point update into a normal surface-correction component and
   a tangent-plane, coverage-preserving component.
2. Train a local controller with train-pair supervision to allocate the two
   components from geometry, residual, and density features.
3. Optimize score-aligned bidirectional CD while preserving correspondence and
   feature regions.

This must be differentiated experimentally from score-based iterative
denoising, latent noise disentanglement, and noise-only adaptive routing:

- Score-Based Point Cloud Denoising, ICCV 2021:
  <https://openaccess.thecvf.com/content/ICCV2021/html/Luo_Score-Based_Point_Cloud_Denoising_ICCV_2021_paper.html>
- Denoising Point Clouds in Latent Space via Graph Convolution and Invertible
  Neural Network, CVPR 2024:
  <https://openaccess.thecvf.com/content/CVPR2024/html/Mao_Denoising_Point_Clouds_in_Latent_Space_via_Graph_Convolution_and_CVPR_2024_paper.html>
- Routing on Demand: DSNet for Efficient Progressive Point Cloud Denoising,
  CVPR 2026:
  <https://openaccess.thecvf.com/content/CVPR2026/html/Cheng_Routing_on_Demand_DSNet_for_Efficient_Progressive_Point_Cloud_Denoising_CVPR_2026_paper.html>

Any publication claim requires separate ablations on the complete validation
protocol, per-category analysis, runtime reporting, and qualitative point-cloud
visualization. It must not rely on validation-set training or repeated
validation-driven tuning.
# Baseline continuation (train_90 only, 1 extra epoch; full 100-shape validation)

- Starting checkpoint: `experiments/pgd_normalcorr_continue400_from7908/pgd-shapenet-epoch00-loss4.53743574.npz` (same 79.11 baseline).
- Training: unchanged InfoCD + normal-correction settings, lr `5e-7`, 400 steps, 8-GPU manual sync with global batch 8; only `train_90.txt` was read.
- Full validation result: score `78.27`, CD `65.13`, P2S `91.42`.
- Directional CD: pred→GT `3.587277e-5`; GT→pred `3.216665e-5`; mean CD `6.803942e-5`.
- Decision: below the original full-validation baseline 79.11, so this continuation checkpoint is rejected and is not merged or retained.

# Baseline continuation (train_90 only, 1 extra epoch, lr=1e-7; full 100-shape validation)

- Starting checkpoint: `experiments/pgd_normalcorr_continue400_from7908/pgd-shapenet-epoch00-loss4.53743574.npz`.
- Training used unchanged InfoCD + normal-correction settings, 8-GPU manual sync, global batch 8, and only `train_90.txt`.
- Full validation: score `78.26`, CD `65.13`, P2S `91.39`.
- Directional CD: pred→GT `3.584028e-5`; GT→pred `3.215224e-5`; mean CD `6.799252e-5`.
- Decision: below 79.11 baseline; checkpoint rejected and not retained.
# 后处理均匀化筛选（2026-07-27）

依据 `idea.png` 的“推理阶段轻量均匀化”建议，新增 `tools/postprocess_tangent_uniform.py`：对预测点的 KNN 排斥向量投影到局部 PCA 切平面，避免明显法向偏移；不使用 GT 或验证集训练。使用已有 `scorecheck_2syn20_normalcorr400` 的 20 个固定验证样本作快速筛选（基线 score=79.81，CD=67.47，P2S=92.15）：

- `strength=0.005, iters=1`：score=79.81，CD=67.47，P2S=92.14，基本无变化；
- `strength=0.1, iters=1`：score=79.78，CD=67.53，P2S=92.03，CD 略升但总分下降。

结论：当前轻量切平面排斥未带来总分提升，暂不合入主流程；筛选输出目录已清理，仅保留脚本供后续复现。上述结果仅作 20 样本筛选，不能替代全量验证。

# GT→Pred 覆盖 + 局部密度约束（进行中，2026-07-27）

本轮从原始 79.11 验证集最佳权重继续训练，仅读取 `train_90.txt`，不使用
验证集训练。配置为 `loss_clean_cd_weight=0.05`、
`loss_anti_cluster_weight=0.003`、`anti_cluster_k=8`、
`anti_cluster_margin=0.90`，其余主流程损失保持不变，训练 160 steps。
实验目录：`experiments/clean_coverage_control/`。

通过标准：必须在全量验证集官方 50k mesh 协议下超过 79.11；同时记录
score、CD/P2S、pred→GT 和 GT→pred。未超过基线的 checkpoint 不合入主流程。

## GT→Pred 覆盖 + 局部密度约束结果（2026-07-27）

使用 8 卡、CUDA 12.8 完成全量 100 个验证样本官方 mesh 评测。结果为
score `78.25`，CD `65.12`，P2S `91.39`；平均 CD_pred=`6.799e-5`，
CD_noisy=`2.254e-4`，P2S_pred=`1.020e-5`，P2S_noisy=`1.375e-4`。
方向性 CD 为 pred→GT `3.582202e-5`、GT→pred `3.216320e-5`。

相较原始最佳权重验证集 score `79.11`、CD `66.44`，总分和 CD 均下降，
因此该 checkpoint 不合入主流程，作为淘汰实验产物保留评测记录。

## 原始最佳权重同协议复核（2026-07-27）

为排除评测协议差异，使用原始权重
`experiments/pgd_normalcorr_continue400_from7908/pgd-shapenet-epoch00-loss4.53743574.npz`
按本轮完全相同的 CUDA12.8、50k mesh、seed=8200、patch_batch_size=64、
两阶段 PGD 和 refine-gate 参数复跑全量验证。复核结果为 score `78.25`、
CD `65.12`、P2S `91.39`；方向性 CD 为 pred→GT `3.582106e-5`、
GT→pred `3.216364e-5`（mean CD `6.798470e-5`）。

因此，当前这套推理参数下基线实际为 78.25；此前记录的 79.11 来自不同
的验证/推理配置，不能与本轮 78.25 直接混用。后续候选必须在同一协议下
相对该复核基线比较。

## CD 主要问题归因与后续目标（2026-07-27）

对 `baseline_recheck_cuda128` 的 100 个样本做了密度与方向性 CD 分析：

- mean pred→GT=`3.582106e-5`，mean GT→pred=`3.216364e-5`，前者高约
  11.4%；93% 的样本 pred→GT 更大，说明主要矛盾是系统性的预测离面/位移
  误差，而不是少数异常样本。
- 随机 14 个样本的 pred 1NN 均值约 `0.002872`，GT 约 `0.002665`，pred
  反而整体略稀；occupied voxel 数也没有显示整体过密。因此“点云一簇簇”
  更可能来自局部位移场不连续、patch hard-select seam 与局部离面偏移，不能
  直接用强排斥或全局增密处理。
- 当前 patch fusion=select，每个输出点只保留一个 patch 预测，点数没有因
  overlap 增加；但 patch 边界可能造成位移不连续。

后续主目标确定为：以一阶段贴面结果为锚，二阶段优先降低法向离面误差，
仅施加小幅切向重分布（建议 tangent scale `0.2–0.5`、normal scale
`0–0.15`），并对切向位移做局部幅值门控，避免牺牲 P2S。所有候选仍须在
同一全量验证协议下比较 score 与双向 CD。

## Stage2 surface gate round 固定 26 样本筛选（2026-07-27，进行中）

四个方向均基于同一 baseline checkpoint，仅训练/替换二阶段 gate；验证使用
固定 `holdout_screen_26.txt`、50k mesh、CUDA12.8、seed=8200 的统一协议，
不使用验证集训练。固定集基线为 score `77.73`、CD `63.68`、P2S `91.77`，
方向性 CD 为 pred→GT `4.1905e-5`、GT→pred `3.8494e-5`。

前三个方向已完成 26/26：

| 方向 | score | CD | P2S | pred→GT | GT→pred |
|---|---:|---:|---:|---:|---:|
| control | 77.68 | 63.27 | 92.09 | 3.7263e-5 | 3.3922e-5 |
| plane | 77.68 | 63.27 | 92.09 | 3.7266e-5 | 3.3922e-5 |
| plane_tangent | 77.67 | 63.26 | 92.08 | 3.7279e-5 | 3.3914e-5 |
| plane_tangent_seam | 77.69 | 63.69 | 91.70 | 4.2046e-5 | 3.8372e-5 |
| anti_cluster_0.05_60step | 77.70 | 63.27 | 92.12 | 3.9665e-5 | 3.6143e-5 |
| anti_cluster_0.20_60step | 77.70 | 63.27 | 92.12 | 3.9661e-5 | 3.6143e-5 |
| anti_cluster_0.10_60step | 77.70 | 63.27 | 92.12 | 3.9662e-5 | 3.6144e-5 |

相对固定集基线，三者总 score 暂未提升约 0.05，但双向原始 CD 均明显下降，
说明前三个 gate 方向的原始双向 CD 有改善，但官方 score 未提升；seam 方向
的 score 77.69、CD 63.69、P2S 91.70，相对固定集基线仍略降，且双向 CD
总和 8.0418e-5 略高于基线 8.0399e-5。因此当前四个方向均不合入主流程，
下一步转向新的 CD 优化方法探索（重点处理点聚集/覆盖不均），并继续以完整
验证集 score 及 pred→GT、GT→pred 双向 CD 作为筛选标准。

### 后处理快速否定结果（2026-07-27）

在 control 的 26 样本 holdout 上做 GT-free 局部 PCA 后处理：`project_wlop`
（k=16, strength=0.05, alpha=0.03, repulsion=0.03）双向 CD 总均值约
`7.56e-5`，`tangent_repulsion`（k=8, margin=0.85, step=0.15）约
`7.49e-5`，均劣于 control 的 `7.12e-5`，故不合入。后续重点验证 overlap
weighted fusion、density-aware gate 及 anti-cluster 训练。

最终 anti-cluster 三组固定 holdout 结果均已完成：三种权重的官方结果均为
score `77.70`、CD `63.27`、P2S `92.12`，方向性 CD 也几乎相同；相对固定集
基线 score `77.73`、CD `63.68`，总 score 下降 `0.03`，故不合入。它们的
原始双向 CD 约 `7.5803–7.5808e-5`，虽优于基线 `8.0399e-5`，但未转化为
官方总分提升。

### 本轮最终结论

已获得官方 score 的所有训练/推理方向均未超过固定集基线 77.73：control/plane
约 77.68，plane_tangent 77.67，seam 77.69，anti-cluster 三组 77.70；因此
当前不修改主流程。WLOP/tangent-repulsion 后处理在原始双向 CD 快速筛选中已
退化，weighted fusion 因 Jittor/CUTT 并发编译未产生有效预测，均不纳入最终候选。

### Anti-cluster 短训筛选（2026-07-27）

仅使用训练集、从同一 baseline 继续 60 steps 的三组 checkpoint 已生成：
权重 `0.05/0.10/0.20` 对应训练 loss `1.053011/1.031738/1.025364`。
anti005（权重 0.05）已完成固定 holdout：score `77.70`、CD `63.27`、P2S
`92.12`，pred→GT `3.9665e-5`、GT→pred `3.6143e-5`，双向 CD 总和
`7.5808e-5`。相对基线原始双向 CD 明显下降，但官方总 score 低 `0.03`，
因此暂不合入。anti020 已完成且与 anti005 几乎一致（score `77.70`、CD
`63.27`、P2S `92.12`，双向 CD 总和 `7.5803e-5`），仍未超过基线；anti010
已完成且同样为 score `77.70`、CD `63.27`、P2S `92.12`，双向 CD 总和
`7.5806e-5`；三种 anti-cluster 权重均未提升官方总 score，均不合入主流程。

## 统一协议最佳权重与 CD 提分路线（2026-07-28）

本节固化 2026-07-28 对现有完整验证结果的复核、CD 归因和下一轮执行方案。
除非特别标注，以下验证结果均使用同一套可复现协议：

- `datalist/validate.txt` 的 100 个 shape；
- 从 mesh 采样 50,000 个 clean 点；
- 每个样本的 Gaussian noise std 独立采样自 `[0.005, 0.020]`；
- seed `8200`；
- `patch_size=1500`、`seed_k=7`、`seed_k_alpha=10`；
- select patch fusion、一次 denoise；
- two-stage PGD，`second_stage_scale=0.5`；
- refine gate，`refine_gate_scale=0.25`；
- `/home/starter_code/evaluate.py` 官方 CD/P2S scorer。

### 当前结果口径

在工作区已有、完成全部 100 个样本且使用上述相同协议的结果中，当前最高
权重为：

`experiments/score82_plain_infocd_lr/lr5e6/pgd-shapenet-epoch00-loss4.52375959.npz`

| 权重/结果口径 | CD | P2S | score |
|---|---:|---:|---:|
| 当前同协议完整验证最佳 | `65.35` | `91.68` | **`78.52`** |
| 原最佳提交权重，同协议 seed=8200 复评 | `65.12` | `91.39` | `78.25` |
| 原最佳提交权重，历史本地验证记录 | `66.44` | `91.78` | `79.11` |
| 原最佳提交权重，平台测试 | `68.07` | `91.16` | `79.61` |

当前最佳的原始均值为 CD_pred `6.741e-5`、CD_noisy `2.254e-4`、
P2S_pred `9.79e-6`、P2S_noisy `1.375e-4`。完整日志位于
`experiments/score82_plain_infocd_lr/final_once_lr5e6/evaluate.log`。

历史 `79.11` 来自另一轮随机 clean/noise realization；它不是 seed=8200
协议下的可直接比较基线。后续实验必须在相同 clean、noisy、seed 和 scorer
下做 paired comparison，不能再用 `79.11` 否定或接受 seed=8200 的候选。
当前 `78.52` 权重尚无平台测试成绩。

### CD 方向分解

当前最佳 100-shape 验证预测的 scorer-consistent 双向平方 CD 为：

| CD 方向 | 均值 | 占总 CD |
|---|---:|---:|
| pred→GT | `3.539619e-5` | `52.5%` |
| GT→pred | `3.201600e-5` | `47.5%` |
| total | `6.741219e-5` | `100%` |

pred→GT 略高，但 GT→pred 也占接近一半，不能把问题简化为单一离面误差。
模型同时存在预测点位置误差和 clean surface 覆盖/局部分布误差。

相对原最佳提交权重的同协议复评，当前最佳的 pred→GT 只下降 `1.19%`，
GT→pred 只下降 `0.46%`，总原始 CD 下降 `0.84%`。1000-step、`5e-6`
纯 InfoCD continuation 已有正收益，但仍属于小幅微调，继续重复同类短训的
边际收益预计很低。

### 对应点法向/切向诊断

验证噪声由 clean 点逐点加 Gaussian noise 得到，因此 noisy、clean 和
denoised 天然保留一一对应。对每个 shape 固定抽取最多 1000 个对应点，并用
clean 16-NN PCA 法向分解 paired squared error，得到：

| paired error | noisy | denoised | 变化 |
|---|---:|---:|---:|
| 法向 | `1.689132e-4` | `6.170393e-5` | `-63.5%` |
| 切向 | `3.396806e-4` | `3.448137e-4` | **`+1.5%`** |
| total | `5.085938e-4` | `4.065176e-4` | `-20.0%` |

结论是当前网络主要学会把点拉回连续 mesh 表面，但没有恢复原始 clean
采样的切向位置；切向 paired error 甚至略微增加。高 P2S 与相对低 CD
可以同时出现：P2S 只要求点贴近连续表面，而离散 CD 还会惩罚沿表面滑动、
局部聚集、覆盖空洞和点分布错位。

### 训练目标与 scorer 不对齐

当前最佳 continuation 的关键配置是：

- `pgd_composite_loss=false`；
- `pgd_use_normal_corr_loss=false`；
- `loss_score_relative_weight=0`；
- 实际优化目标为 plain Jittor InfoCD。

plain InfoCD 使用双向最近邻欧氏距离，不利用已有的一一对应，也没有直接
对齐官方的双向平方 CD 和逐样本
`1 - CD_pred / CD_noisy` 得分公式。因此它允许预测点在表面切向滑动到另一
个近邻位置，且原始距离目标会让高噪声样本贡献更大的梯度。

后续不能只在参数中打开 `pgd_use_normal_corr_loss`：只有
`pgd_composite_loss=true` 时，normal/tangent correspondence、relative CD
和 score-relative CD 才进入有效 objective。

### 噪声强度失配

当前 PGD `execute` 虽接收 `noise_std` 和 `category_id`，但 backbone 实际
调用为 `self.feature_nets(pcl_noisy, None, offset)`，即没有使用噪声强度。
整个 `[0.005, 0.020]` 范围共用一个去噪映射。

按实际 noise std 分桶后的完整验证结果为：

| noise std | 样本数 | CD | P2S |
|---|---:|---:|---:|
| `<0.00875` | 31 | **`51.09`** | `88.96` |
| `[0.00875,0.01250)` | 22 | `64.29` | `92.26` |
| `[0.01250,0.01625)` | 20 | `73.50` | `93.32` |
| `>=0.01625` | 27 | `76.55` | `93.13` |

低噪声是最主要的 score 尾部。官方 scorer 对每个样本先除以自身 noisy
baseline；低噪声样本分母更小，固定的绝对残差 floor 会被放大。下一轮必须
同时解决“模型不感知 sigma”和“训练 objective 未做 per-sample relative
normalization”两个问题。

### 类别与结构尾部

当前 validate split 的两个类别结果为：

| synset | 样本数 | CD | P2S |
|---|---:|---:|---:|
| `03642806` | 48 | `61.45` | `90.82` |
| `04074963` | 52 | `68.95` | `92.49` |

`03642806` 的 CD 低 `7.50` 分。薄片、边缘、铰接面和相邻表面更容易让
1500 点局部 patch 产生表面归属错误及切向漂移。该差异支持后续加入可靠的
多尺度上下文和 normal/tangent stage-2 gate，但它们应排在 objective 对齐
与 noise conditioning 之后。

## 下一轮提分执行顺序

### P0：score-aligned correspondence objective

第一优先级是在当前 `78.52` 权重上做充分训练的 objective 对齐实验，而
不是继续增加推理后处理。起始配置建议为：

```text
--pgd_composite_loss
--pgd_use_normal_corr_loss
--normal_corr_normal_weight 2.0
--normal_corr_tangent_weight 1.0
--loss_corr_weight 1.0
--loss_relative_weight 0.5
--loss_pred_cd_weight 0.25
--loss_clean_cd_weight 0.5
--loss_score_relative_weight 0.5
--loss_infocd_weight 0.05
--loss_uniform_weight 0.02
```

设计目的：

1. paired normal/tangent Huber 直接恢复 synthetic clean 对应点，阻止表面
   切向漂移；
2. pred→GT 与 GT→pred 分开约束，避免只改善 P2S；
3. score-relative CD 按每个样本 noisy baseline 归一化，提升低噪声权重；
4. InfoCD 降为辅助几何项；
5. uniform 保持很弱，避免破坏薄结构、边缘和 P2S。

该配置只作为 train-holdout sweep 起点，不直接宣称最佳。必须记录每个
weighted loss term 的实际数值及梯度贡献，避免再次出现正则项权重非零但
贡献低几个数量级、训练结果与 control 无差异的问题。

此前 300–1000 step 的多种 composite/DCD/Sinkhorn continuation 多数变化
接近 control。下一轮应使用 warmup-cosine 和足以改变权重的训练长度；先在
固定 train holdout 上比较学习曲线，再决定是否扩大到数千 step 或多 epoch。

### P1：真正的 noise conditioning

在 P0 得到稳定收益后，将标量 noise std 编码后注入 Jittor PGD：

1. 用小型 MLP 生成 noise embedding；
2. 以 FiLM、feature bias/scale 或 residual gate 的形式注入 backbone；
3. stage 1 与 stage 2 使用可分开的 sigma-conditioned gate；
4. 训练时使用已知 synthetic sigma；
5. 推理时使用现有 local noise estimator，并在训练中加入 estimator 误差
   扰动，降低 train/test conditioning gap。

快速原型可先使用 low/mid/high 三档 gate；验证有效后再改成连续条件化。
不再测试全样本统一的 `pred_weight<1`，因为既有 sweep 已证明全局缩放会
同时损害 CD 和 P2S。目标是让低噪声分支学习更精确的残差，而不是简单减少
所有位移。

### P2：薄结构和 patch 一致性

仅当 P0/P1 后 `03642806` 仍显著落后时，依次测试：

1. 给 stage 2 增加较大 receptive field 的多尺度局部上下文；
2. 将 stage-2 位移显式拆成 normal/tangent 两部分；
3. 用局部平面置信度、边缘置信度和 sigma 共同控制两个 gate；
4. 对同一点在 overlap patches 中的 displacement 加归一化 consistency；
5. 只在高置信平面区域启用轻量 tangent redistribution。

既有 weighted fusion、displacement consensus 和硬 PCA 后处理均未稳定提升
完整 score，不能作为这一阶段的默认实现。

### P3：覆盖损失只作残余修正

paired correspondence 本身已经是一一对应的强覆盖监督。只有当 P0/P1 后
GT→pred 仍明显落后，才加入：

- balanced local transport；
- normalized tangent-spacing consistency；
- 低权重、单边 anti-collapse penalty。

不重复现有短程 DCD/Sinkhorn/anti-cluster 配方。所有 coverage loss 必须先
确认有效 contribution，并同时检查 P2S、薄结构和 pred→GT，不能只看原始
GT→pred 均值。

## 目标门槛与实验选择

当前 score 为 `78.52`，P2S 为 `91.68`。若 P2S 保持不变：

| 目标总分 | 所需 CD | 约需降低 CD residual |
|---|---:|---:|
| `80.00` | `68.32` | `8.5%` |
| `82.00` | `72.32` | `20%` |

每个新 round 遵守以下选择流程：

1. 只在固定、category-stratified 的 train holdout 上选择配置；
2. 缓存相同 50k clean/noisy 输入，保证候选与 control paired；
3. 至少分别报告 low/mid/high sigma、类别、pred→GT 和 GT→pred；
4. 优先选择 official total score 提升且 P2S 不明显回退的候选；
5. 完整 100-shape validate 只运行该 round 的唯一 holdout winner；
6. 不用 validate 反复调权重、early stop 或构建类别词表；
7. 训练、loss、模型、评测和 CUDA 路径全部保持 Jittor。

第一阶段的成功门槛是固定 train holdout 上可复现地提升总分至少 `0.3`，
同时 low-noise CD 有明确改善、P2S 回退不超过 `0.1`。达到该门槛后才消耗
一次完整 validate；完整验证目标先超过同协议 `78.52`，再推进 `80+`。

## 2026-07-28 按优先级实测结果

本轮严格执行“总 score 必须提高”的选择规则。固定 screen 使用训练集拆出的
26 个 shape，所有分支复用同一批 50k clean/noisy 点云和官方 scorer。该
screen 上当前完整配对基线为：

| 分支 | CD | P2S | score |
|---|---:|---:|---:|
| 当前 `78.52` 权重的固定 train-holdout 基线 | `63.98` | `92.09` | **`78.03`** |

基线方向 CD 为 pred→GT `4.1346014e-5`、GT→pred `3.8159719e-5`，
总和 `7.9505733e-5`。

### P0：score-aligned correspondence objective（拒绝）

在当前最佳权重上训练 300 steps，warmup-cosine、峰值 LR `1e-5`。除普通
plain InfoCD continuation 外，分别测试：

- 直接 score-relative / 双向 relative CD；
- relative normal/tangent correspondence，法向/切向权重 `2/1`；
- 更强调切向 correspondence 的法向/切向权重 `1/2`。

原始 correspondence 量级远小于总 loss，因此实现了按每个样本 noisy
correspondence baseline 归一化的新 loss；其数值从 `1e-4` 量级变为约
`0.7–0.9`，确认不再是无效项。相关 Jittor 单测通过。

完整 26-shape 结果为：

| 分支 | CD | P2S | score | pred→GT | GT→pred |
|---|---:|---:|---:|---:|---:|
| baseline | `63.98` | `92.09` | `78.03` | `4.1346014e-5` | `3.8159719e-5` |
| plain control | `63.98` | `92.09` | `78.03` | `4.1344937e-5` | `3.8159974e-5` |
| score-only objective | `63.98` | `92.09` | `78.03` | `4.1346889e-5` | `3.8157691e-5` |
| relative corr `0.25` | `63.98` | `92.09` | `78.03` | `4.1347726e-5` | `3.8158266e-5` |
| tangent-weighted corr | `63.98` | `92.09` | `78.03` | `4.1346986e-5` | `3.8157734e-5` |

所有方向差异都只有约 `1e-9`，没有总 score 增益，P0 全部拒绝。结论是
仅改变当前 backbone 的训练 objective，即使 loss 数值已对齐，也未在
300-step gate 中产生可用的预测变化；不能因训练 loss 下降而继续扩大训练。

### P1：sigma-conditioned stage gate（拒绝）

PGD 原先接收 `noise_std`，但未实际使用。本轮加入零初始化的小型 Jittor
MLP，分别控制 stage 1 和 stage 2 residual；开启时旧 checkpoint 初始输出
保持完全一致。只训练 4 个 conditioner 参数组，共 300 steps。

训练后 stage 1 gate 保持 `1.0`；score-gate 的 stage 2 gate 随 sigma 为：

| sigma | `0.005` | `0.00875` | `0.0125` | `0.01625` | `0.020` |
|---|---:|---:|---:|---:|---:|
| stage-2 gate | `0.8330` | `0.8863` | `0.9081` | `0.9094` | `0.9019` |

这说明模型确实学到条件化，而不是单一全局缩放。现有局部 PCA sigma
estimator 在固定 train holdout 上排序相关性约 `0.83`，但高噪声有系统性
低估；线性校准 `true≈1.992141×raw−0.004548` 将 MAE 从约 `0.00417`
降到 `0.00205`。估计器的逐邻域 eigensolve 已改为数学等价的批量计算。

总 score 结果不合格：

| 分支/配对范围 | CD | P2S | score | 对应基线 |
|---|---:|---:|---:|---:|
| score gate，15 shape | `64.48` | `92.25` | `78.37` | `78.42` |
| corr gate，6 shape | `60.35` | `91.96` | `76.15` | `76.20` |

score gate 同时降低 CD 和 P2S；corr gate 与其预测效果近似。二者不可能达到
`+0.3`，因此分别在 15/6 个严格配对样本处提前停止。P1 说明仅按 sigma
缩放完整 stage-2 residual 会牺牲贴面精度，且估计器误差会进一步削弱条件化。

### P2：stage-2 normal/tangent dual gate（拒绝）

使用无 GT 的 16-NN 局部几何将 stage-2 residual 分成法向/切向部分，并
训练零初始化的 per-point 双门控。只训练 4 个 gate 参数组，共 300 steps。
最终训练 batch 的平均法向 gate 为 `0.9387`，切向 gate 为 `0.4181`；
模型确实重点压制了切向 residual。

但官方总分更差：

| 分支/配对范围 | CD | P2S | score | 对应基线 |
|---|---:|---:|---:|---:|
| dual gate，4 shape | `60.33` | `89.57` | `74.95` | `75.25` |

即 CD `60.37→60.33`、P2S `90.13→89.57`、总 score `−0.30`，两项都
下降，故在 4 个样本处提前停止。原因是用局部 chord 估计的 normal/tangent
分解在曲面、薄结构和边缘并不可靠；“估计切向”中仍含有效法向去噪分量，
强抑制后直接损害 P2S 和总 score。

### 本轮选择结论

本轮没有候选满足：

1. 固定 holdout 总 score 至少 `+0.3`；
2. P2S 回退不超过 `0.1`；
3. CD 改善足以覆盖 P2S 损失。

因此没有运行新的完整 100-shape validate。当前同协议完整验证最佳仍是：

`experiments/score82_plain_infocd_lr/lr5e6/pgd-shapenet-epoch00-loss4.52375959.npz`

对应 CD `65.35`、P2S `91.68`、score **`78.52`**。

P3 的 balanced transport、Sinkhorn、DCD、tangent spacing、anti-cluster
和 overlap consistency 在前序相同 train-holdout 流程中已经完成多轮实测，
均没有稳定提升总 score，本轮不重复消耗完整评测。

下一轮若继续，应优先解决两个前置问题，而不是继续加 loss：

1. 用训练集 full-shape noisy cloud 训练/标定可泛化的 sigma estimator，
   先把推理 sigma MAE 明显降到 `<0.0015`；
2. 用可靠的局部曲面 frame（需要比 chord-cross 更稳的法向与边缘置信度）
   做 normal/tangent gate，并对 gate 加 P2S-preserving 约束；任何候选先在
   4–8 个配对样本上同时提高 CD、P2S 和总 score，再扩大到 26 shape。

## 2026-07-28 结构改造续轮

本轮继续使用同一固定 train-only paired holdout，并保持“CD 与最终 score
必须同时提高”的硬约束。26-shape 基线仍为 CD `63.98`、P2S `92.09`、
score `78.03`；6-shape 早筛子集基线为 CD `60.35`、P2S `92.05`、
score `76.20`。

### A1：StraightPCF-style velocity × remaining distance（拒绝）

在当前 PGD encoder/decoder 上加入 patch-level remaining-distance head，
并使用 noisy-clean 随机插值轨迹训练。distance head 末层为零初始化，
因此加载旧 checkpoint 时 `exp(log_distance)=1`，初始预测完全不变。

只训练 distance head 的 26-shape 结果：

| 训练步数 | CD | P2S | score | pred→GT | GT→pred |
|---:|---:|---:|---:|---:|---:|
| baseline | `63.98` | `92.09` | `78.03` | `4.1346014e-5` | `3.8159719e-5` |
| 100 | `63.77` | `91.69` | `77.73` | `4.2103e-5` | `3.7969e-5` |
| 300 | `63.72` | `91.61` | `77.67` | `4.2179300e-5` | `3.7949196e-5` |

两个训练长度都略微改善 GT→pred 覆盖，却显著恶化 pred→GT 和 P2S；
训练更久恶化更明显。当前问题不是单个全局/patch-level 去噪距离不足，
而是不同点、不同局部结构需要不同方向和幅度。A1 不进入完整验证。

### A2：6000 context → 1500 core fusion（拒绝）

网络接收每个 seed 最近的 6000 点作为上下文，但只融合最近 1500 个 core
点，输出点数与基线完全一致。实际 6-shape 结果：

| 分支 | CD | P2S | score |
|---|---:|---:|---:|
| baseline | `60.35` | `92.05` | `76.20` |
| context 6000 / core 1500 | `58.76` | `90.60` | `74.68` |

CD/P2S/score 分别下降 `1.59/1.45/1.52`。单样本 batch=1 推理约
`16.2 min`；batch=4 仍约 `10–16 min`，而标准 1500-point patch 约
`3.3 min`。原因是 backbone 只在 1500 点 patch 上训练，直接改变输入
规模会改变多层下采样、邻域和特征统计，形成严重 train/test context
shift。若以后重试多尺度，必须在训练阶段加入同样的 core-context sampler，
不能只改推理。

### A3：learned normal + signed surface-distance head（拒绝）

在 Stage-2 decoded feature 上增加 per-point normal 与 signed distance
head；distance 末层零初始化，surface correction 初始为零。训练只使用
train split clean patch 估计的局部法向，并加入 sign-invariant normal
alignment 与 relative point-to-plane loss。相关 21 个 Jittor 单测通过，
CUDA 训练与 checkpoint 保存正常。

6-shape 官方结果：

| 分支 | CD | P2S | score |
|---|---:|---:|---:|
| baseline | `60.35` | `92.05` | `76.20` |
| learned surface head, 100 steps | `60.36` | `92.03` | `76.19` |

该 head 将 CD 提高 `0.01`，但 P2S 下降 `0.02`，最终 score 下降 `0.01`，
不满足双提高。方向 CD 显示它仍主要改善 GT→pred 覆盖，同时让少量预测点
离面；因此不通过 26-shape gate。

### A4：non-shared Stage-2 head（拒绝）

Stage-2 使用从旧共享 backbone 完整复制的独立网络，旧 checkpoint 的
全部参数可一一映射，step 0 输出保持不变。早筛只训练 Stage-2 的最终
displacement MLP，并在 decoded feature 处停止梯度，避免无意义地反传
整条复制 backbone。先跑 100 steps；只有固定 6-shape 上 CD 与 score
同时提高才扩到 26 shape。

100-step 的 6-shape 官方结果为 CD `60.35`、P2S `91.99`、score
`76.17`。CD 与 baseline 持平，P2S/score 分别下降 `0.06/0.03`，
因此仅训练 Stage-2 输出 head 不足，不进入扩大评估。

### Oracle：当前预测附近确实存在双改善表面方向

为区分“附近没有更优解”和“网络没有学到更优方向”，在同一固定 6-shape
train holdout 上做了只用于诊断的 mesh closest-point oracle。它使用与官方
P2S 相同的 PCU mesh 最近点查询，将当前预测向 GT mesh 最近点插值；该过程
读取 GT mesh，**不能用于正式验证或提交**。

| mesh 投影比例 | CD | P2S | score |
|---:|---:|---:|---:|
| `0.00` | `60.35` | `92.05` | `76.20` |
| `0.10` | `61.53` | `93.56` | `77.55` |
| `0.30` | `63.61` | `96.10` | `79.86` |
| `0.50` | `65.27` | `98.01` | `81.64` |
| `1.00` | `67.22` | `100.00` | `83.61` |

结论是当前预测附近存在很大的 CD/P2S 双改善空间，主要瓶颈是剩余离面误差，
而不是必须先做切向覆盖。后续结构顺序因此固定为：先学可靠 surface
correction，再做 tangent transport，最后用 edge confidence 限制薄结构和
尖锐边缘上的错误切向移动。

### A5：direct surface-residual vector head（拒绝）

A5 用零初始化的 3D residual head 替代 A3 的 `normal × signed distance`，
避免法向/距离的符号耦合。训练监督只来自 train split clean 点和法向，
推理仍只输入 noisy point cloud。

第一版使用 paired clean normal residual，100-step 结果为 CD `60.36`、
P2S `92.06`、score `76.21`，仅比 baseline 三项各高 `0.01`。训练诊断显示
归一化 residual loss 始终约为 `1.0`，head 基本学成零校正。将有界输出从
`0.02*tanh(raw)` 改为零点单位斜率的
`0.02*tanh(raw/0.02)` 后，loss 仍无法稳定低于零输出基准，说明冻结特征
缺少逐点剩余表面误差的可预测信息，而非只有梯度缩放问题。

第二版用当前预测在 clean patch 中的最近点近似 oracle，并只保留最近点法向
分量。它在训练 patch 上能将 residual ratio 降到约 `0.98`，但 6-shape
官方结果显著下降到 CD `59.81`、P2S `91.77`、score `75.79`。有限 clean
采样的 nearest point 不是连续 mesh closest point，量化/对应误差会被 head
学入并伤害泛化。因此 A5 不继续增加步数。

### A6：完整 non-shared Stage-2 backbone（当前正向候选）

Stage1 保持原最佳 checkpoint；Stage2 的 encoder、decoder、codebook 和
displacement head 从原共享 backbone 完整复制，再以 `5e-6` 小学习率单独
训练。step 0 与 baseline 完全一致，训练不使用 validate GT。

固定 6-shape 官方结果：

| 分支 | CD | P2S | score | 相对 baseline |
|---|---:|---:|---:|---:|
| baseline | `60.35` | `92.05` | `76.20` | — |
| A6, 100 steps | `60.42` | `92.05` | `76.24` | CD `+0.07`，score `+0.04` |
| A6, 300 steps | `60.49` | `92.04` | `76.27` | CD `+0.14`，score `+0.07` |

A6 是本轮第一个随训练长度增加而持续提高 CD 和总 score、且 P2S 基本不变
的结构候选，但尚未达到 `+0.3` 的扩大门槛。暂不盲目延长到 600 steps，
下一步在 A6-300 缓存预测上加入固定、无 GT 的 surface/tangent 几何修正。

### 后续改进主线：surface stabilization → tangent transport → edge confidence

该主线作为后续优先方向固化：

1. **Surface stabilization**：以 A6 完整独立 Stage2 为基础，优先保持或
   提高 P2S；禁止使用验证 mesh。若继续学习 surface residual，训练目标应
   来自更密集的训练表面采样或训练 mesh closest point，而不是 1500-point
   patch 内离散 nearest neighbour。
2. **Tangent transport**：先复用已在 26-shape train holdout 上验证过的
   固定配置：局部 PCA 法向回投 strength `0.15`，随后 `k=8`、
   margin `0.85`、step `0.125` 的单步 tangent repulsion。它只使用预测点云
   和 noisy cloud，不改变点数。当前正在 A6-300 的固定 6-shape 输出上评估。
3. **Edge confidence**：若固定 transport 有正增益，再从局部协方差特征构造
   置信度，例如 `planarity=(λ2-λ1)/(λ3+eps)`、曲率
   `λ1/(λ1+λ2+λ3)`、法向邻域一致性及 noisy/pred 图稳定性。只在高
   planarity、高法向一致性的平滑区域开放 transport；边缘、薄片、尖角和
   高曲率区域将步长压到零或仅允许更保守移动。
4. **学习式 transport**：固定置信度通过后，再训练零初始化的 per-point
   tangent step/gate；输入必须使用旋转不变量或局部 frame 表达，loss 同时
  约束 clean→pred coverage、pred→surface fidelity 与 transport magnitude。
5. **Context 一致性**：最后实现训练期 `6000-context/1500-core` sampler 和
   跨 patch overlap consistency，消除 A2 的 train/test context shift。不能
   再只在推理时放大 context。

统一选择门槛保持不变：固定 train-only holdout 上 CD 与最终 score 必须同时
提高，P2S 不得实质退化；早筛目标为 score 至少 `+0.3`。只有通过该门槛的
候选才扩到 26-shape，随后才允许一次完整 100-shape validate。
