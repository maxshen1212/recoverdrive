# Run: PRAM-R-style baseline
- Plan: docs/superpowers/plans/2026-11-22-pramr-baseline.md (approved by the user in plan mode, 2026-11-22)
- Decisions by the user: one reliability score per sensor (not per modality); one test-split run, all 10 faults,
  seeds 0-2.

## Status
Done. Results in `REPORT.md` §4.2 (local) and the report Doc; numbers in `report_figures/tables.md` (S1, S2, S5) and
`report_figures/bootstrap.md`.

## Checkpoints
| Step | Commit | Verification | Notes |
|---|---|---|---|
| 1-4 implementation | 98dc0a8 | 194 passed, ruff clean; golden tests pin the old A1/A2 outputs; 1643/1643 logged `test_final` corruptions rebuild identically; replay check 4830/4830 | `pramr.py`, `PramR`, A2 flip for reliability answers, A1 `missing_field` fallback, reference/tune scripts, test guard |
| 5 reference ranges | 88c3775 | 243 clean dev keyframes | 78.2 % of clean frames have at least one indicator outside p5-p95 (recorded, not tuned) |
| 6 smoke | - | `smoke_pramr`: 78 frames, 0 errors, 0 invalid answers | |
| 7 dev screen | - | `dev_pramr_s0`: 42 runs, 1701 frames, 0 errors, 0 invalid; flags identical to `vlm_screen_s0`; replay 1701/1701 | |
| 8-9 tuning and freeze | aaef26f | rule fixed in the plan: highest F1 with false alarm <= 0.10; chosen: mosaic, theta_cam 400, theta_lidar 750, delta 50 (dev F1 0.352, FA 0.095); `dev_pramr_s0_frozen` all cached, identical metrics | text-only never below FA 0.11 |
| 10 test run | aaef26f (clean tree) | `test_pramr`: 120 runs, 4830 frames, 0 errors; flags identical to `test_final` 4830/4830; replay 4830/4830 | killed once at 102/120 runs with no error in the log; resumed under the same tag and commit (finished runs kept, 18 partial runs restarted from frame 0 with cached answers) |
| 11 report | (this commit) | bootstrap: 55/55 existing rows unchanged, 10 new; `tables.md` sections before S1 unchanged | |

## Results (test)
- All faults, mean ± std over 3 seeds: PRAM-R F1 0.344 ± 0.020 (Ours-full 0.765), false alarm 0.116 (0.121), missed
  fallback 0.161 (0.053), recovery 0.118 (0.587).
- Sensor faults only: PRAM-R F1 0.308 vs 0.714; Ours-full − PRAM-R F1 +0.406 [0.372, 0.436].
- Recall per fault: L4 1.00, C3 0.44, L1 0.05, C1/C2/C4/L2 0.00. Agent faults: 3 of 720 frames recovered.

## Notes
- The `e2e-vlm` branch (end-to-end VLM exploration of 2026-11-22) is not merged; its results are dev-only trend
  tests and stay out of the report.
- /data was full (0 bytes free, filled by other jobs) when this was committed; only the repo's own `__pycache__` and
  `.ruff_cache` were removed to make room.
