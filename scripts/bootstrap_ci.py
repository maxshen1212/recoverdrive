"""Block bootstrap 95 % intervals for E1/E2 metrics and paired differences (FutureWork Step 1).

Usage:
    uv run python scripts/bootstrap_ci.py --tags test_final test_checks_only \
        --methods B1 B2 B3 ours_detect ours_full checks_only --reference ours_full --out report_figures/bootstrap
A tag can carry a prefix (TAG=PREFIX) that renames its methods to PREFIX:<name>, e.g. test_abl_no_checks=nochecks.
"""

import argparse
import json
from pathlib import Path

from recoverdrive.bootstrap import bootstrap_metrics, merge_runs
from recoverdrive.metrics import markdown_table
from recoverdrive.runner import git_state

REPO = Path(__file__).resolve().parent.parent


def load(tag: str) -> list[dict]:
    recs = []
    for f in sorted((Path("outputs/eval") / tag / "runs").glob("*.jsonl")):
        recs += [r for r in (json.loads(line) for line in f.read_text().splitlines() if line.strip())
                 if "error" not in r]
    if not recs:
        raise SystemExit(f"no records under outputs/eval/{tag}/runs")
    return recs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True, help="run tags, optionally TAG=PREFIX")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--reference", default="ours_full")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output path stem (writes .md and .json)")
    args = p.parse_args()

    runs = []
    for spec in args.tags:
        tag, _, prefix = spec.partition("=")
        runs.append((load(tag), prefix or None))
    records = merge_runs(runs)
    missing = [m for m in args.methods if not any(m in r["methods"] for r in records)]
    if missing:
        raise SystemExit(f"methods without any output: {missing}")
    rows = bootstrap_metrics(records, args.methods, args.reference, args.n_boot, args.seed)
    n_frames = len(records)
    text = "\n".join([
        f"# Bootstrap 95 % intervals ({n_frames} frames; tags {', '.join(args.tags)})", "",
        (f"Blocks = fault episodes and clean gaps, resampled within (run fault, fault/clean); {args.n_boot} replicates; "
         "seeds pooled, so point estimates can differ slightly from the mean-over-seeds tables. "
         f"Rows '{args.reference} - X' are paired differences."), "",
        markdown_table(rows, ["method", "metric", "point", "lo", "hi"])])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(text + "\n")
    out.with_suffix(".json").write_text(json.dumps({"git": git_state(REPO), "args": vars(args), "frames": n_frames,
                                                    "rows": rows}, indent=1))
    print(text)


if __name__ == "__main__":
    main()
