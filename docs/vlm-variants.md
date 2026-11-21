# VLM-centric variants of the diagnosis agent

Run: `docs/superpowers/runs/2026-11-21-finish-and-vlm-variants.md`. Code: `VLMVariant` in
`src/recoverdrive/methods.py`, prompts at the end of `src/recoverdrive/vlm.py`, tests in `tests/test_variants.py`.

## Why

Two earlier results show that, in the frozen system, the VLM adds little of its own:
- **Step 1 (test split):** on sensor faults, checks-only and Ours-full have the same F1 (0.714). The VLM confirms the
  deterministic flags.
- **Falsification replays:** Ours-full still violates 212 of the 214 worst counterexamples of the checks.

The frozen prompt tells the VLM "The deterministic checks are reliable" and asks it to use the images only to
confirm a flagged sensor, so it follows the checks, including their false alarms and misses. The falsification
triage puts the remaining errors in the checks' evidence: false alarms on clean frames, faults below the detection
limit, and cameras with fewer than two reference objects (assumption A3). A VLM that judges the raw images itself
could, in principle, cover these cases.

## The variants

The goal was to let the VLM reason more and to use less hand-made engineering. Each variant changes only the
diagnosis agent. The recovery controller, the 3-sample vote, the shield, the model (Qwen3-VL-8B-Instruct) and the
inputs (camera mosaic of the current and previous frame, LiDAR top view, check report) are the same as Ours-full.
Every variant has a `_detect` output (first answer, no recovery) and a `_full` output (with recovery).

| Variant | Calls per frame (before the vote) | What the deciding call sees | Idea |
|---|---|---|---|
| Ours (frozen) | 1 | images + check report; "the checks are reliable" | the VLM confirms the flags |
| `think` | 1 | images + check report; "the checks can be wrong" | reason step by step before the verdict |
| `think_nochecks` | 1 | images only | the same reasoning, no deterministic evidence |
| `perceive` | 2 | its own notes + check report + images | stage 1 describes every sensor from the raw data (no checks); stage 2 weighs the notes against the report |
| `perceive_nochecks` | 2 | its own notes + images | two stages, VLM only |
| `percam` | 8 | per-sensor notes + check report + current mosaic and top view | stage 1 looks at each camera alone at 960×540 with its previous image (6 calls) and at the LiDAR top views with the point count (1 call); stage 2 decides |

Output of the deciding call (JSON schema): `analysis` (free reasoning, written first), `verdict` (ok, uncertain or
fault:<sensor>), `fault_type`, `confidence`, `evidence`. Stage-1 calls are not diagnosis calls: the agent faults
A1–A3 hit only the deciding call, as for Ours. The prompts were written once and not tuned (deferred question D3);
one smoke run on one dev scene checked only the output format.

## Protocol

- Dev split only (6 scenes). A new variant on the test split needs the user's approval (spec §5.4: tune on dev,
  freeze, run test once).
- Screening: all variants and the references (B0, checks-only, Ours-detect, Ours-full) in one run over the same
  frames: sensor faults C1–C4, L1, L2, L4, seed 0 (`outputs/eval/vlm_screen_s0`).
- The best variants then run with all 10 faults and seeds 0–2, and replay the falsification counterexamples.

## Results

(filled in when the runs finish)
