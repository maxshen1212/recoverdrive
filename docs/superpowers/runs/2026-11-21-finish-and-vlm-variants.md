# Run: finish Phase B, VLM-centric variants, report and release
- Plan: docs/superpowers/plans/2026-11-21-finish-and-vlm-variants.md
- Goal condition: "跑完剩下沒跑完的結果，跑完後接著最佳化我們之前討論的想辦法讓VLM思考的更多，最小化我們做的engineering，最大化的去發揮VLM的能力，像是可以做各種排列組合嘗試：直接用VLM先抓特徵，在結合pipeline固定抓的數據，最後所有數據再進VLM綜合分析一次產出結果等...反正盡可能在合理的次數嘗試。最後分析完結果後，把全部的實驗結果整理起來生出數據圖、demo影片、實驗變因對照表等有用的圖表、系統資料流方塊圖，更新Google Docs上面的報告、數據，並且merge分支到main，然後push。https://drive.google.com/drive/folders/1iJn5UZNgCUXQtAErjc7FvjrqkfstP44a?usp=drive_link"
- Started: 2026-11-21 03:15 (server gvl2)
- pre-authorized: "最佳化...各種排列組合嘗試" (new method variants, dev split only); "更新Google Docs上面的報告、數據"
  (the report Doc and the Drive folder above); "merge分支到main，然後push".
- Continues `docs/superpowers/runs/2026-11-20-course-requirements.md` (Phase B tasks B3–B7).

## Status
Current: T4 screening restarted at 04:19 on both vLLM servers (GPU 4 port 8002, GPU 7 port 8001; finished
frames replay from the cache); `think_trust` (D7) runs in parallel as `vlm_screen_s0_trust`. T1 done. Next: replay of
the counterexamples with the best variant (T5), then figures and report.

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
| T2a | (this commit) | triage of `gpu_v1`: 209 violating trials, 0 bug candidates; confirmations 145/147 and 67/67 still violated with Ours-full | falsification log, Phase B sections |
| T1a | (this commit) | 4 CARLA campaigns complete: 160 runs (summary.md 40 each), all exit 0 | resumed with `--resume`; confirmation running |
| T2b | (this commit) | CARLA triage: 29 R4 + 1 R5 violation (1 collision), all undetected C2 rotations; 0 bug candidates | falsification log, CARLA section |
| T1 | (this commit) | CARLA confirmation: 10 replays with Ours-full, exit 0; the collision scenario also collides with Ours-full | B5 complete |

## Drive
- Folder `experiment_result/report_figures_2026-11-21` (`1BBO-rIVZXMCK_T9hkz1oK9UplcJ6PGsM`).
- Demo video `recoverdrive_demo_c1_pedestrian.mp4` (`16TnNjXg6KOq0jPG1VVI-2WtfMhUDf2w7`, 7,020,097 bytes), made by
  `scripts/make_demo_video.py` from the recorded E3 demo runs.
| T8a | (this commit) | Doc phase 1 applied in 4 guarded batches, 0 errors; read back: Formal Requirements (§3), Figure 2 data flow (§5), experiment table (§6 Setup), Verification and Falsification with 2 tables and Figure 8, 7. Assurance Case with Figure 12, captions renumbered (3–7, 11), 8. Conclusion; PDF export 21 pages checked | images from GitHub raw URLs at `cceadb6` (branch pushed) |

## Deferred questions
- D3 (T3): variant prompts are written once, not tuned (one smoke check for format errors only).
- D4 (T5): the 2 variants with the best pooled `_full` F1 on dev seed 0 (fallback errors within 0.02 of Ours-full)
  get seeds 1–2.
- D6 (T4): the screening covers the 7 sensor faults only (one vLLM server; no second GPU was free at 03:30). The
  agent faults A1–A3 test the re-ask and vote, which all variants share with Ours-full; they run for the top
  variants in T5. If you prefer otherwise: add A1–A3 to the screen (about +40% VLM time).
- D7 (T5): plan change. The partial screen showed every variant below Ours-full in recall. `think` changes two
  things at once (reasoning first, and "the checks can be wrong"), so T5 adds the clean ablation `think_trust`
  (frozen prompt + reasoning field) on the same frames, and replays the falsification counterexamples with the
  most promising variant. Seeds 1–2 run only if a variant beats Ours-full on seed 0; otherwise block-bootstrap
  intervals over fault episodes on seed 0 measure the gap. If you prefer otherwise: 3 seeds for the best 2
  variants anyway (about 4–5 h of VLM time on one shared GPU).
- D8 (T8): Doc images come from raw.githubusercontent.com at the pushed commit, because the Docs API inserts images
  only from a public URL and the server has no Drive client for large files. The new figures are therefore in
  `docs/media/` of the public repo (data figures were local-only before). If you prefer them private: remove them
  from the repo after the Doc is final (the Doc keeps its own copy of each image).
- D5 (T4–T5): the variants run on the dev split only. A test-split run of a new variant needs your OK (spec §5.4).

## Blockers
(none)
