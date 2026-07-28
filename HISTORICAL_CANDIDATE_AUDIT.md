# Historical validation-candidate audit

The fixed reference is the complete 100-shape validation result (`79.11`,
CD `66.44`, P2S `91.78`).  The entries below are **screening evidence only**:
their `evaluate.log` files contain 20 shapes, so they must not be treated as
full-validation improvements.  They are ordered by proxy score and should be
re-run on all 100 validation shapes with the identical 50k protocol before
selection.

| Candidate output | 20-shape score | CD | P2S | Priority |
|---|---:|---:|---:|---|
| `scorecheck_2syn20_refinegate_rotconsistency05_100` | 80.42 | 68.36 | 92.49 | high |
| `scorecheck_2syn20_context125` | 80.42 | 68.36 | 92.48 | high |
| `scorecheck_2syn20_refinegate_detach200` | 80.42 | 68.35 | 92.48 | high |
| `scorecheck_2syn20_local_refine_gate_highgain600` | 80.42 | 68.35 | 92.48 | high |
| `scorecheck_2syn20_local_refine_gate200` | 80.42 | 68.35 | 92.48 | high |
| `scorecheck_2syn20_blend_gate_ref50` | 80.40 | 68.34 | 92.46 | medium |
| `scorecheck_2syn20_blend_base_gate75` | 80.40 | 68.33 | 92.47 | medium |
| `scorecheck_2syn20_context600` | 80.39 | 68.34 | 92.44 | medium |
| `scorecheck_2syn20_twostage1700_pca_k16_s0p10` | 80.39 | 68.19 | 92.59 | high (CD) |
| `scorecheck_2syn20_twostage1700_pca_k24_s0p15` | 80.39 | 68.18 | 92.59 | high (CD) |
| `scorecheck_2syn20_coverage_cd_clean40_uniform06` | 80.30 | 68.20 | 92.39 | high (CD) |
| `alpha_twostage1700_a100` | 80.30 | 68.13 | 92.48 | same source; one rerun |
| `alpha_twostage1700_a105` | 80.18 | 67.98 | 92.38 | same source; optional |
| `alpha_twostage1700_a095` | 80.04 | 67.91 | 92.17 | same source; optional |

The `pred_weight=1.1` branch is especially important despite its lower proxy
score: `scorecheck_2syn20_refinegate_predweight11` gives score `79.91`, CD
`67.78`, P2S `92.03`, with directional means pred→GT
`3.6094069e-5` and GT→pred `3.2240151e-5`.  Its re-audit gives `79.24` on
the same 20-shape subset, so both records are proxy-only and the branch should
be included in the next complete-100 run.

The remaining 79.3–79.44 blend/decoder-distribution rows are also 20-shape
proxies, but are lower priority and largely duplicate the same source outputs.
No candidate in this audit is being rejected for being below 80; only the
validation-set baseline comparison is relevant.

## Coverage-tail short-screen note

The temporary `coverage-tail` loss implementation was smoke-trained with
5,000-point inputs only.  Its 26-shape 5k starter-evaluator runs produced
scores around `8.3` (P2S around `0.7`), which is a protocol mismatch rather than
evidence against the loss: the valid baseline gate is the 50k mesh protocol.
Those short-screen checkpoints and predictions were removed after recording
this result; the train-only implementation remains available for a correctly
matched 50k experiment if resources permit.
