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

Later commits (`528b367..14d3024`) changed only documentation, comments, the model download script, the CARLA
launcher, and demo recording in `run_e3.py`. No change affects any number above.

## Notes

- **Rebased commit.** `docs/superpowers/runs/2026-11-03-phase5.md` lists P5-T6 as `1890796`. That commit was
  rewritten by a rebase and is now `528b367`. Every commit in the table above is an ancestor of `main`
  (checked with `git merge-base --is-ancestor`).
- **Reproduce one result:** `git checkout <commit>`, then run the command from `REPORT.md` §10. The SQLite caches in
  `outputs/cache/` hold the SAM 3 and VLM results, so a rerun with the same cache returns the same answers without
  calling the models again.
- **Not in the backup:** `data/nuscenes` (5.1 GB; download again with `./scripts/download_nuscenes_mini.sh`),
  model weights (Hugging Face), and the uv environments.
