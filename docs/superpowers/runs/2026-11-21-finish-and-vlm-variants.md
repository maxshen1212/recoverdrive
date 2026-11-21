# Run: finish Phase B, VLM-centric variants, report and release
- Plan: docs/superpowers/plans/2026-11-21-finish-and-vlm-variants.md
- Goal condition: "跑完剩下沒跑完的結果，跑完後接著最佳化我們之前討論的想辦法讓VLM思考的更多，最小化我們做的engineering，最大化的去發揮VLM的能力，像是可以做各種排列組合嘗試：直接用VLM先抓特徵，在結合pipeline固定抓的數據，最後所有數據再進VLM綜合分析一次產出結果等...反正盡可能在合理的次數嘗試。最後分析完結果後，把全部的實驗結果整理起來生出數據圖、demo影片、實驗變因對照表等有用的圖表、系統資料流方塊圖，更新Google Docs上面的報告、數據，並且merge分支到main，然後push。https://drive.google.com/drive/folders/1iJn5UZNgCUXQtAErjc7FvjrqkfstP44a?usp=drive_link"
- Started: 2026-11-21 03:15 (server gvl2)
- pre-authorized: "最佳化...各種排列組合嘗試" (new method variants, dev split only); "更新Google Docs上面的報告、數據"
  (the report Doc and the Drive folder above); "merge分支到main，然後push".
- Continues `docs/superpowers/runs/2026-11-20-course-requirements.md` (Phase B tasks B3–B7).

## Status
Current: T1 running (4 CARLA campaigns on GPU 7 since 03:20); T3 done. Next: T4 dev screening (vLLM on GPU 4).

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
| T3 | (this commit) | 175 passed, ruff clean; replay check 4830/4830 on `test_final` | `VLMVariant` (5 variants), prompts in `vlm.py`, `tests/test_variants.py`; resume keeps unreached rows |

## Deferred questions
- D3 (T3): variant prompts are written once, not tuned (one smoke check for format errors only).
- D4 (T5): the 2 variants with the best pooled `_full` F1 on dev seed 0 (fallback errors within 0.02 of Ours-full)
  get seeds 1–2.
- D5 (T4–T5): the variants run on the dev split only. A test-split run of a new variant needs your OK (spec §5.4).

## Blockers
(none)
