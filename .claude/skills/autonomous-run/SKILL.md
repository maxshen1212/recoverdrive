---
name: autonomous-run
description: Use when running a RecoverDrive phase, plan, or experiment unattended - under /goal, when told to work through a plan end to end without check-ins, or when resuming such a run after a pause, /goal re-arm, or context compaction.
---

# Autonomous run

## Overview
The user is away while a /goal loop drives you. Make steady, recoverable progress. Write the plan yourself and execute it. Commit a checkpoint after each verified task. Defer minor questions to one batch at the end. Stop only for problems that need the user's decision.

The /goal evaluator sees only the conversation and cannot run commands. Every claim of progress needs its evidence in your text, for example "T3 verified: 52 passed, ruff clean, commit a1b2c3d".

## Start or resume
1. Look in `docs/superpowers/runs/` for a run log for this target. If one exists, read it and continue from its Status section. Never redo checkpointed tasks.
2. If there is none, read CLAUDE.md and the spec (`docs/superpowers/specs/2026-11-03-recoverdrive-design.md`). Then write the plan to `docs/superpowers/plans/YYYY-MM-DD-<slug>.md` following superpowers:writing-plans conventions.
   - Do not use EnterPlanMode or ExitPlanMode. Nobody is there to approve.
   - Every task needs a verification command whose output proves the task is done.
3. Some plan decisions are not settled by the spec. For each one, pick the option most consistent with the spec, mark it in the plan as `DEFAULT (D<n>)`, and log it as a deferred question. If the decision would change the spec (see Severe below), follow the BLOCKED procedure before writing any code.
4. Create the run log from the template below.

## Per-task loop
1. Implement the task.
2. Verify. Run the task's verification command, `uv run pytest -q`, and `uv run ruff check src tests scripts`. State the results in your text.
3. Checkpoint.
   - Update the run log: add a checkpoint row and update Status.
   - Run `git add -A && git commit` with the message `checkpoint(<slug>): T<n> <task title>`, a 1–3 line body, and the Co-Authored-By trailer from the current attribution reminder.
   - Never commit while tests fail. Never push.
4. Go to the next task.

## Classify every problem
Ask: "If I take the reasonable default now and the user later disagrees, what is lost?"

**Severe: stop and ask now (BLOCKED procedure) if any of these is true:**
- A core assumption of the plan or spec is shown to be wrong. For example, a signal the method depends on does not separate faults, or a dataset or model cannot do what the plan needs. As a result, the remaining tasks would be wasted or need restructuring.
- The fix would change spec §1.4 decisions, the §5.1 method list, the §5.2 metrics, or the §5.4 evaluation rules. This includes using test scenes before the user freezes results.
- The action is irreversible or outward-facing: deleting data or results, pushing, force operations, installing system-level software, using credentials, or downloading more than 10 GB to `/`.
- The same blocker survives 3 different fix attempts. Use superpowers:systematic-debugging first.
- No GPU has had enough free memory for 30 minutes.
- Undoing the default later would cost more than redoing one task.

**Pre-authorized: the goal condition explicitly allows this exact action or decision.** Examples are "may freeze and run test once after Phase 3 calibration" and "may install CARLA under /data/maxshen". Proceed within the stated scope, and log it in the run log as `pre-authorized: <quote>`. Anything outside the quoted scope is still Severe; for example, CARLA needing sudo when only an install under /data/maxshen was authorized.

**Minor: everything else.** Choose the reasonable default, log it as a deferred question, and continue. Typical cases:
- threshold or parameter choices within the plan's latitude
- a library API that differs from the plan
- an extra helper or test
- reordering or merging tasks
- an ambiguous metric detail that has a sensible default
- a plan step that turns out to be unnecessary

## BLOCKED procedure
1. Leave the tree safe.
   - Commit verified work as a checkpoint.
   - Stash half-done changes with `git stash -u -m "blocked: <slug> T<n>"` and record the stash name in the run log.
2. Stop the vLLM server and any GPU jobs you started.
3. Record the blocker in the run log.
4. Send a one-line PushNotification.
5. End the turn with a message that starts with `BLOCKED:`. It must contain:
   - what you found, with evidence
   - which severe rule applies
   - 2–3 options, with your recommendation
   - what you will do after each answer

   Make no tool calls after this message.

## Finish
When all tasks are checkpointed and the target's completion criteria pass:
1. Run the full verification again and state the results.
2. Stop vLLM. Show `nvidia-smi` output proving your GPU memory is released.
3. Send a PushNotification.
4. Write the final report in the user's language. It contains:
   - what was done
   - a checkpoint table with commit hashes
   - key results
   - deferred questions D1..Dn, each with the issue, the choice you made, and what changes if the user chooses differently
   - a question: should the commits be pushed?

## Project rules during a run
- Run `source scripts/env.sh` in every shell.
- Run `nvidia-smi` before each GPU job and pick a card with enough free memory. The GPUs are shared.
- Use dev scenes only (`configs/splits.yaml`). Never edit the spec; propose spec changes as deferred questions or BLOCKED.
- Background work: /goal skips evaluation while background work is running. Short jobs with `run_in_background` are fine because they finish. Long-lived servers must run detached:
  `setsid nohup env CUDA_VISIBLE_DEVICES=<gpu> VLM_GPU_UTIL=0.5 ./scripts/serve_vlm.sh > outputs/logs/vllm.log 2>&1 &`
  Then poll `http://127.0.0.1:8000/health`. To stop it: `pkill -INT -u "$USER" -f "[v]llm serve"` (`-u` spares other users' vLLM servers on the shared GPUs; the brackets stop pkill from matching your own shell, whose command line contains the pattern).
- Plans and run logs live in `docs/superpowers/` and are tracked: they go into the checkpoint commits.

## Run log template
```markdown
# Run: <target>
- Plan: docs/superpowers/plans/<file>.md
- Goal condition: <verbatim>
- Started: <date time>

## Status
Current: T<n> - <state>. Next: <step>.

## Checkpoints
| Task | Commit | Verification | Notes |
|---|---|---|---|

## Deferred questions
- D1 (T<n>): <issue>. Chose: <default>. If you prefer otherwise: <what changes, cost>.

## Blockers
(none)
```

## Common mistakes
| Mistake | Instead |
|---|---|
| Stopping to ask about a minor choice | Log a deferred question and continue |
| Guessing through a severe problem | BLOCKED procedure |
| Saying "done" without evidence in text | State the test counts, commit hash, and metrics |
| Leaving vLLM running at BLOCKED or finish | Stop it and show `nvidia-smi` |
| Redoing work after a resume | Read the run log Status first |

## /goal condition template
Give this to the user to fill in `<target>`, `<pre-authorizations>`, and `<criteria>`:
```
/goal 使用 autonomous-run skill 完成 <target>。事先授權：<pre-authorizations>。對話中要出現以下證據：所有 plan 任務都有 checkpoint commit；`uv run pytest -q` 全過且 `uv run ruff check src tests scripts` 沒有錯誤；<criteria>；vLLM 已關閉；最後一次列出所有 deferred questions。或者當 Claude 送出以 BLOCKED: 開頭的訊息等我決定時就停止。或在 80 個 turn 後停止並回報進度。
```
