# Results provenance

Which commit produced each reported result. `REPORT.md` and `outputs/` are not tracked; a verified backup of them
(with this file and a `git bundle` of the repo) is on the lab server at
`/data/maxshen/backups/recoverdrive/2026-11-15/` (`recoverdrive-results-2026-11-15.tar.gz`, checksum in `SHA256SUMS`).
Each result commit also has an annotated tag `results/*` (last column).

- HEAD when the backup was made: `14d3024dc528f69d54f147ce9e9c72bd3f039157`
- All experiments ran on 2026-11-03 (server-local time) on the shared lab server (4 x RTX 6000 Ada).

## Which commit produced which result

"Recorded" = the commit is written in the run's `meta.json`. For the test split, `scripts/run_eval.py` refuses to
start unless the config is frozen and the git tree has no uncommitted changes to tracked files, so these runs match
the commit exactly. "Inferred" = the script does not record a commit; the commit is matched from file timestamps
and the git log, and the note says how sure that is.

| Experiment | Result path (under `outputs/`) | Commit | Source | Tag |
|---|---|---|---|---|
| **E1/E2 main table** (test, B0-B3, Ours-detect, Ours-full, 10 faults, 3 seeds, 120 runs) | `eval/test_final/` | `5d5283cc86aa3c66c67c7bd95ab669517cfa4c68` | Recorded. The frozen commit (thresholds and prompt calibrated on dev, then frozen). | `results/e1e2-test` |
| E5 offline recovery ablations (replay of the `test_final` logs: no vote / no re-observe / no isolate, VLM gate) | `eval/test_final/ablation_replay.{json,md}` | `5ea57bbf06019807a6352376438e737b82964621` | Inferred. Written 17:08; the replay script was committed in `5ea57bb` at 17:09. Replay only re-reads the frozen `test_final` logs. | `results/e5-replay` |
| E5 no consistency checks (test) | `eval/test_abl_no_checks/` | `b0c0eb0900ef2b9df150b692790011f75066b18e` | Recorded. | `results/e5-no-checks` |
| E5 SAM 3 -> Qwen grounding (test) | `eval/test_abl_qwen_grounding/` | `2b86d9071a76b00f9e8615b2d26a661379f41f19` | Recorded. | `results/e5-qwen-grounding-4b` |
| E5 Qwen3-VL-4B-Instruct (test) | `eval/test_abl_4b/` | `2b86d9071a76b00f9e8615b2d26a661379f41f19` | Recorded. | `results/e5-qwen-grounding-4b` |
| E5 Qwen3-VL-8B-Thinking (test, seed 0 only) | `eval/test_abl_thinking/` | `d0cabf39fbb837d98b16c9996a1da244d8a285fb` | Recorded. | `results/e5-thinking` |
| **E3 CARLA closed loop** (4 routes x clean/C1 x 3 seeds x B0/B1/Ours-full, 72 runs) | `e3/e3_final/` | `dd1f5266a3ba5609f21f53557d47535667b70cdf` | Inferred. The run started at 08:15 on HEAD `6069963`; its results were committed in `dd1f526` (11:40). Between the two commits only `scripts/run_e3.py` changed (world cleanup, per-process ports, `--summarize`, error handling); `src/` and `configs/` are identical, so the method code is the same in both. `run_e3.py` does not record a commit or check for a clean tree. | `results/e3-carla` |
| **E4 latency** | `e4/latency.{json,md}`, built from the dev runs `eval/e4_ours/` and `eval/e4_baselines/` | runs: `4d32c3613c4d314491857e9dcd7b1e4bc386b07e`; report: `b0c0eb0900ef2b9df150b692790011f75066b18e` | Inferred. Dev runs record `commit: null` by design. They ran 17:44-17:50 on HEAD `4d32c36`; `latency.json` was written 18:09 on HEAD `b0c0eb0` (the E4 checkpoint). | `results/e4-latency` |
| Dev tuning, Phases 1-3 (prompt versions, calibration, smoke runs) | `eval/p1`, `p2`, `p3_dev`, `prompt_*`, `t5_check`, `t6_check`, `smoke_*`; `calib/`; `e0/` | before `5d5283c` | Dev runs record `commit: null`. Not used for any reported test number. | - |
| Report figures and tables | `report_figures/` (repo root, not tracked) | `make_figures.py` as of `2b86d90` | Inferred. Written 23:27. Later commits changed only a docstring in `make_figures.py`. | - |
| `REPORT.md` | (repo root, not tracked) | text written at the P5-T6 checkpoint `528b367afbcf6133ad69e912e6cc2ea2cb59468a` | Last edited 23:57 on HEAD `14d3024`. | - |
| CARLA demo GIFs | `e3/demo/` (copies tracked in `docs/media/`) | `529fcf6` | From the git log. | - |

### Step 1 results (2026-11-15, branch `step1-report-completeness`)

Every run below records its commit and `dirty: false` in its own `meta.json` / output JSON (`git_state`).

| Result | Path (under `outputs/` unless noted) | Commit | Source |
|---|---|---|---|
| Checks-only on dev (seeds 0–2, all faults; Step 2 classifier training data) | `eval/dev_checks_only/` | `8f49275` | Recorded. Flags identical to `p3_dev` on 2430/2430 frames. |
| **Checks-only on the test split** (3 seeds, all faults) | `eval/test_checks_only/` | `3483460` | Recorded. Flags identical to `test_final` on 4830/4830 frames. |
| 4B + VLM gate (offline replay of the 4B test run) | `eval/test_abl_4b/ablation_replay.{md,json}` | `ca7a200` | Recorded (replay reproduces 4830/4830). |
| E4 re-timed, 8B and 4B (dev scene-0061) | `eval/e4_ours_8b_v2/`, `eval/e4_ours_4b/`; `e4_8b_v2/`, `e4_4b/` | `3483460` | Recorded. |
| Bootstrap intervals | `report_figures/bootstrap.{md,json}` (repo root, not tracked) | `3483460` | Recorded in the JSON. |
| Step 1 tables S1–S5 | `report_figures/tables.md`, key `step1` of `summary.json` | branch head | `scripts/make_figures.py`; old keys unchanged. |

Tag: `results/checks-only-test` (`3483460`).

### Course requirements, Phase A (2026-11-20, branch `worktree-course-requirements`)

Specs, proofs and CPU-only falsification (plan `docs/superpowers/plans/2026-11-20-course-requirements.md`). Each output
records its commit in its JSON or `meta.json`.

| Result | Path | Commit | Source |
|---|---|---|---|
| R1–R6 on the logged runs (test split, E3) and Clopper–Pearson bounds | `report_figures/requirements.{md,json}` (repo root, not tracked) | `38707da` | Recorded. Reads `eval/test_final`, `eval/test_checks_only`, `e3/e3_final`. |
| Shield and controller verification (S1–S7, C1–C6) | `report_figures/formal.{md,json}` | `3c3f707` | Recorded. |
| nuScenes falsification, CPU faults (C1, C4, L1, L2, L4; 2,400 trials, checks-only, dev) | `falsify/nuscenes_cpu_v1/` | `fbd56ff` | Recorded (`meta.json`, `summary.md`). |
| Falsification triage | `falsify/nuscenes_cpu_v1/triage.{md,json}` | `20ebeb8` | Recorded. |
| Assurance case figure | `docs/media/assurance_case.png` (tracked) | `20ebeb8` | `scripts/make_assurance_figure.py` from the files above. |

Tags: `results/requirements-logs` (`38707da`), `results/formal-proofs` (`3c3f707`), `results/falsification-nuscenes-cpu` (`fbd56ff`).

### Course requirements, Phase B, and VLM-centric variants (2026-11-21, server gvl2)

Run log `docs/superpowers/runs/2026-11-21-finish-and-vlm-variants.md`. Server gvl2 (8 x RTX 6000 Ada, driver 565):
CUDA 12 builds of the same package versions (`b4d5408`); the SQLite cache was copied from the first server, so cached
SAM 3 and VLM answers are reused. "dirty" below means only run-log files were uncommitted at launch.

| Result | Path (under `outputs/`) | Commit | Source |
|---|---|---|---|
| nuScenes falsification, C2/C3 (960 trials, checks-only, dev) | `falsify/nuscenes_gpu_v1/` | `89d95d7` | Recorded. |
| Triage of the C2/C3 campaigns | `falsify/nuscenes_gpu_v1/triage.{md,json}` | `dfd2846` | Recorded. |
| Replays of the worst counterexamples with Ours-full | `falsify/nuscenes_confirm_{cpu,gpu}_v1/` | `89d95d7` | Recorded. |
| CARLA falsification (4 campaigns, 160 runs) | `falsify/carla_v1_{pedestrian,lead_brake}_{random,tpe}/` | started `89d95d7`; resumed `77bfb9f` (pedestrian TPE: `dfd2846`) | Recorded; `meta.json` holds the resume commit and `resumed_trials`. The scenario and run code did not change between these commits. |
| CARLA replays with Ours-full | `falsify/carla_v1_confirm_{pedestrian,lead_brake}/` | `d3095f1` (dirty), `2b4a032` | Recorded. |
| E3 regression cell on gvl2 | `e3/regress_gvl2/` in the worktree | `89d95d7` | Recorded. |
| VLM-centric variants, screening (dev, seed 0, 7 faults) | `eval/vlm_screen_s0/` | started `dfd2846` (dirty); restarted on `ad13504` | Recorded; `meta.json` holds `ad13504`. Restarted at 04:19 with a second VLM server; finished runs and the cache were reused. Between the two commits only `think_trust` was added; the screened variants' code is identical. |
| `think_trust` ablation (same frames) | `eval/vlm_screen_s0_trust/` | `ad13504` | Recorded. |
| Replays with variants | `falsify/nuscenes_confirm_{cpu,gpu}_v1_{think_trust_full,think_nochecks_full}/` | `df05870` (C1, C4, L1, L2, L4), `648712a` (C2, C3); dirty: docs only | Recorded. |
| Round 2 of the variants: `perceive_trust`, `imgflag`, `imgflag_checks` (dev, seed 0, 7 faults, same frames) | `eval/vlm_screen_s0_round2/` | `9a55dcb` | Recorded. |
| Seeds 1–2: B0, checks-only, Ours-detect/full, `think_trust` | `eval/vlm_trust_s12/` | `9a55dcb` | Recorded. |
| Seeds 1–2: `imgflag`, `imgflag_checks` | `eval/vlm_imgflag_s12/` | `47b6f2b` (dirty: figures script and docs only) | Recorded. |
| Agent faults A1–A3 (dev, seed 0): Ours, `think_trust`, `imgflag` | `eval/vlm_agent_s0/` | `47b6f2b` | Recorded. |
| CARLA replays with `imgflag_checks` (5 lowest-R4 and the lowest-R5 scenario per hazard) | `falsify/carla_v2_imgflag_{r4,r4_ped,r5}/` | `47b6f2b` | Recorded. The pedestrian R4 replays ran in a second process (`r4_ped`); the first process was stopped before them. |
| Figures and tables of this run | `report_figures/results_2026-11-21.{md,json}`, `fig6`–`fig10`; `docs/media/{falsification,vlm_variants,vlm_variants_per_fault,vlm_variants_seeds,vlm_cost,vlm_replays,dataflow,assurance_case}.png` (tracked) | branch head | `scripts/make_results_figures.py`, `make_dataflow_figure.py`, `make_assurance_figure.py` |
| Demo video | Drive `experiment_result/report_figures_2026-11-21/recoverdrive_demo_c1_pedestrian.mp4` | `c8361ce` | `scripts/make_demo_video.py` from `e3/demo/` |

Tags: `results/falsification-nuscenes-gpu` (`89d95d7`), `results/falsification-carla` (`77bfb9f`, the resume commit in `meta.json`), `results/vlm-variants-round1` (`ad13504`), `results/vlm-variants-round2` (`9a55dcb`), `results/vlm-imgflag` (`47b6f2b`).

Later commits (`528b367..14d3024`) changed only documentation, comments, the model download script, the CARLA
launcher, and demo recording in `run_e3.py`. No change affects any number above.

### PRAM-R-style baseline (2026-11-22, branch `pramr-baseline`)

Run log `docs/superpowers/runs/2026-11-22-pramr-baseline.md`. Server gvl2, CUDA 12 vLLM build (`test_final` used the CUDA 13 build of the same versions). The thresholds were tuned on dev only and frozen in `aaef26f` before the single test run.

| Result | Path (under `outputs/`) | Commit | Source |
|---|---|---|---|
| PRAM-R-style baseline, dev screen (2 settings, seed 0, 7 sensor faults) | `eval/dev_pramr_s0/` (+ `pramr_tune.{md,json}`) | `88c3775` | Recorded. Flags identical to `vlm_screen_s0` on 1701/1701 frames; replay of the thresholds exact on 1701/1701. |
| PRAM-R-style, frozen setting on dev | `eval/dev_pramr_s0_frozen/` | `aaef26f` | Recorded. All 1701 answers from the cache; metrics equal the chosen tuning row. |
| **PRAM-R-style baseline on the test split** (3 seeds, all faults) | `eval/test_pramr/` | `aaef26f` | Recorded. Flags identical to `test_final` on 4830/4830 frames; replay exact 4830/4830. The process was killed once at 102/120 runs (no error in the log) and resumed under the same tag and commit; the 18 partial runs restarted from frame 0 with cached answers. vLLM: the CUDA 12 build of 2026-11-21 (`test_final` used the CUDA 13 build of the same versions). Reference ranges: `configs/pramr_reference.yaml` (`88c3775`). |

Tag: `results/pramr-test` (`aaef26f`).

## Notes

- **Rebased commit.** `docs/superpowers/runs/2026-11-03-phase5.md` lists P5-T6 as `1890796`. That commit was
  rewritten by a rebase and is now `528b367`. Every commit in the table above is an ancestor of `main`
  (checked with `git merge-base --is-ancestor`).
- **Reproduce one result:** `git checkout <commit>`, then run the command from `REPORT.md` §10. The SQLite caches in
  `outputs/cache/` hold the SAM 3 and VLM results, so a rerun with the same cache returns the same answers without
  calling the models again.
- **Not in the backup:** `data/nuscenes` (5.1 GB; download again with `./scripts/download_nuscenes_mini.sh`),
  model weights (Hugging Face), and the uv environments.
