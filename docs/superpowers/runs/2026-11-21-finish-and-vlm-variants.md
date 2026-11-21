# Run: finish Phase B, VLM-centric variants, report and release
- Plan: docs/superpowers/plans/2026-11-21-finish-and-vlm-variants.md
- Goal condition: "跑完剩下沒跑完的結果，跑完後接著最佳化我們之前討論的想辦法讓VLM思考的更多，最小化我們做的engineering，最大化的去發揮VLM的能力，像是可以做各種排列組合嘗試：直接用VLM先抓特徵，在結合pipeline固定抓的數據，最後所有數據再進VLM綜合分析一次產出結果等...反正盡可能在合理的次數嘗試。最後分析完結果後，把全部的實驗結果整理起來生出數據圖、demo影片、實驗變因對照表等有用的圖表、系統資料流方塊圖，更新Google Docs上面的報告、數據，並且merge分支到main，然後push。https://drive.google.com/drive/folders/1iJn5UZNgCUXQtAErjc7FvjrqkfstP44a?usp=drive_link"
- Started: 2026-11-21 03:15 (server gvl2)
- pre-authorized: "最佳化...各種排列組合嘗試" (new method variants, dev split only); "更新Google Docs上面的報告、數據"
  (the report Doc and the Drive folder above); "merge分支到main，然後push".
- Continues `docs/superpowers/runs/2026-11-20-course-requirements.md` (Phase B tasks B3–B7).

## Status
Current: round 2 done (T10–T15). vLLM and CARLA stopped (GPUs 3, 4, 7 at 2 MiB). Only open item: D5 (test split,
needs your OK).

Phase B state found at the start (outputs in `/data/maxshen/recoverdrive/outputs/falsify/`):
- B3 done: `nuscenes_gpu_v1`, C2/C3 × R1/R2/R3/R6 × random/TPE, 960 trials (checks_only).
- B4 done: `nuscenes_confirm_cpu_v1` and `nuscenes_confirm_gpu_v1` (ours_full replays of the worst trials).
- B5 regression cell done: `outputs/e3/regress_gvl2` r0 lead_brake C1 ours_full s0: no collision, route completion
  1.0, as in `e3_final` (steps 67 vs 70, stop steps 0 vs 1).
- B5 campaigns stopped early: lead_brake random 10/40, lead_brake TPE 0/40 (its CARLA server did not start in time),
  pedestrian random 8/40, pedestrian TPE 9/40.

## Checkpoints
| Task | Commit | Verification | Notes |
|---|---|---|---|
| T0 | 77bfb9f | 170 passed, ruff clean | plan, run log, CARLA `--resume` / `--confirm` |
| T3 | dfd2846 | 175 passed, ruff clean; replay check 4830/4830 on `test_final` | `VLMVariant` (5 variants), prompts in `vlm.py`, `tests/test_variants.py`; resume keeps unreached rows |
| T2a | 2fd9cfe | triage of `gpu_v1`: 209 violating trials, 0 bug candidates; confirmations 145/147 and 67/67 still violated with Ours-full | falsification log, Phase B sections |
| T6a | c8361ce | tests and ruff at commit time | figure, diagram and video scripts; experiment table |
| T6b | 741de3a | tests and ruff at commit time | assurance figure reads all falsification outputs |
| T5a | d3095f1 | tests and ruff at commit time | `think_trust` ablation variant (D7) |
| T1a/T2b | 2b4a032 | tests and ruff at commit time | CARLA campaigns complete and triaged |
| T6c | ad13504 | tests and ruff at commit time | results figures merge extra tags; bootstrap intervals |
| T1 | d507ced | CARLA confirmation: 10 replays with Ours-full, exit 0; the collision scenario also collides with Ours-full | B5 complete |
| B7 | 339f4ce | tests and ruff at commit time | Phase B numbers in the verification text |
| T8a | a512894 | tests and ruff at commit time | demo video on Drive |
| T6d | cceadb6 | tests and ruff at commit time | final falsification figure and assurance case |
| T8a | 77242e2 | tests and ruff at commit time | report Doc phase 1 |
| T6e | df05870 | tests and ruff at commit time | results figures union several tags for the seed table |
| T6f | 648712a | tests and ruff at commit time | variant figures (comparison and per-fault F1) |
| T4-T7 | 899e1d8 | tests and ruff at commit time | variant results, replays, figures and docs |
| T8 | f567c6f | tests and ruff at commit time | report Doc, provenance Doc and Drive uploads |
| T9 | 0ce577c | `main` fast-forwarded to the branch (the repo's linear-history convention); in the main checkout `uv sync` builds torch 2.14.1+cu126 (CUDA available), 176 passed, ruff clean; replay check 4830/4830; pushed over HTTPS: origin/main eb34fe3..f567c6f | the worktree is kept (it holds `outputs/e3/regress_gvl2`) |
| T10 | 9a55dcb | 180 passed, ruff clean; replay check 4830/4830 | `perceive_trust` (the 3-step flow with the frozen guidance), `imgflag` (VLM image-quality flag + frozen Ours), `imgflag_checks` (VLM flag + rules only); new flag name `vlm_image` |
| T14a | bd13b34 | tests and ruff at commit time | CARLA replays can rank scenarios by R4 (`--by`) |
| round-2 | 47b6f2b | tests and ruff at commit time | round-2 plan (T10–T15) after the stop-hook review |
| T13 | e315700 | `vlm_agent_s0`: 729 frames, 0 errors; 181 passed, ruff clean | recovery 0.661 Ours-full, 0.622 `think_trust`, 0.661 `imgflag`; unnecessary fallback 0.056 / 0.089 / 0.154 |
| T11 | 67f9493 | `vlm_screen_s0_round2`: 42 runs, 1,701 frames, 0 errors; bootstrap vs Ours-full | `perceive_trust` F1 0.617 (gap +0.109 [+0.070, +0.149]); `imgflag` 0.726 (−0.001 [−0.028, +0.025]) with false alarm 0.246 vs 0.089; `imgflag_checks` 0.723 at 1 call/frame |
| T14 | e0a5990 | `carla_v2_imgflag_{r4,r4_ped,r5}`: 12 replays, all finished (the cut-off duplicate of the main R4 process is kept as `pedestrian_confirm.partial-20261121T0805`) | front camera isolated 0/153 fault steps; rear camera isolated on 65/308 clean steps ("frozen"); R4 3/10 coincidental fixes, 1 new violation; collision unchanged |
| T12 | (this commit) | `vlm_trust_s12`, `vlm_imgflag_s12`: 84 runs and 3,402 frames each, 0 frame errors; figures regenerated | 3 seeds: F1 0.729 ± 0.003 Ours-full, 0.726 ± 0.005 `think_trust`, 0.727 ± 0.002 `imgflag`, 0.726 ± 0.003 `imgflag_checks`; dataflow diagram lists the round-2 variants |
| T15 | (this commit) | report Doc read back and PDF export checked; provenance Doc read back; Drive file sizes match; 181 passed, ruff clean | merged into `main` by fast-forward and pushed with this commit |

## Drive
- Report Doc `1TejQjwoDeW_gQMlVCWMl-G1R87ICXqNJk1hCnP4fHA8`, updated in place (title unchanged).
- Provenance Doc `1J8CKk1YbCBAkJppdpPmcgnT_GXlWwc7XE_7zX7381Wk`: Phase A and Phase B/variant sections added.
- Folder `experiment_result/report_figures_2026-11-21` (`1BBO-rIVZXMCK_T9hkz1oK9UplcJ6PGsM`): demo video
  (`16TnNjXg6KOq0jPG1VVI-2WtfMhUDf2w7`, 7,020,097 bytes) and 7 figures.
- `experiment_result/recoverdrive_results_2026-11-21_{summaries,screen_frames_part1,screen_frames_part2,trust_frames}.zip`
  (split because the connector refuses requests over 10 MB).
- Round 2 (16:36 UTC): report Doc updated in place (variants paragraph, 3 table rows, round-2 paragraphs, seeds table,
  Figures 2, 9 and 10 replaced from `docs/media` at `6459ea8`, Abstract, discussion and Conclusion sentences;
  PDF export checked, 26 pages); provenance Doc: 5 round-2 rows. Drive `report_figures_2026-11-21`:
  `fig_vlm_variants_dev_round2.png`, `fig_vlm_variants_per_fault_f1_round2.png`, `fig_vlm_cost_vs_f1_round2.png`,
  `fig_vlm_variants_seeds.png`, `fig_system_dataflow_round2.png`; `experiment_result`:
  `recoverdrive_results_2026-11-21_round2_{summaries,screen_frames,seeds_trust_frames,seeds_imgflag_agent_frames}.zip`.

## Deferred questions
- D3 (T3): variant prompts are written once, not tuned (one smoke check for format errors only).
- D4 (T5): the 2 variants with the best pooled `_full` F1 on dev seed 0 (fallback errors within 0.02 of Ours-full)
  get seeds 1–2.
- D6 (T4): the screening covers the 7 sensor faults only (one vLLM server; no second GPU was free at 03:30). The
  agent faults A1–A3 test the re-ask and vote, which all variants share with Ours-full. Closed in round 2: A1–A3 ran
  for Ours-full, `think_trust` and `imgflag` (`vlm_agent_s0`); see D10 for `imgflag_checks`.
- D7 (T5): plan change. The partial screen showed every variant below Ours-full in recall. `think` changes two
  things at once (reasoning first, and "the checks can be wrong"), so T5 added the clean ablation `think_trust`
  (frozen prompt + reasoning field) on the same frames, and replayed the falsification counterexamples with the
  most promising variants; block-bootstrap intervals over fault episodes measured the gaps. Closed in round 2: seeds
  1–2 ran for the references, `think_trust` and both image-flag variants (5,103 frames per method); the seed-0
  conclusions hold (same F1 within ±0.02, the variants trade false alarms for missed fallbacks).
- D8 (T8): Doc images come from raw.githubusercontent.com at the pushed commit, because the Docs API inserts images
  only from a public URL and the server has no Drive client for large files. The new figures are therefore in
  `docs/media/` of the public repo (data figures were local-only before). If you prefer them private: remove them
  from the repo after the Doc is final (the Doc keeps its own copy of each image).
- D5 (T4–T5): the variants run on the dev split only. A test-split run of a new variant needs your OK (spec §5.4).
- D9 (round 2): the exploration stops after round 2 (9 variants). The variants now cover reasoning first (with and
  without the frozen guidance), no checks at all, the three-step flow (VLM features → checks → final VLM call, in three
  forms), per-sensor views, and the VLM as a feature extractor for the pipeline. More variants would mean tuning prompts
  on the same 6 dev scenes, with no held-out data to check them on (the test split needs D5). If you prefer otherwise:
  name the next variant; each screen costs about 1–1.5 h of VLM time on 3 GPUs.
- D10 (T13): the image check of `imgflag*` is a feature call, so the agent faults A1–A3 do not reach it (as for the
  stage-1 calls of round 1). `imgflag_checks` has no other VLM call and is therefore left out of the agent-fault table.
  If you prefer otherwise: apply A1/A3 to the image check (an invalid or late answer then means "no VLM flag", so the
  method falls back to checks-only) and define a flip for A2 (ok ↔ abnormal per camera); about 30 lines and one run.

## Blockers
(none)
