"""Closed-loop falsification in CARLA: random vs. guided (TPE) search over scenario parameters for runs that violate
R5 (TTC braking and no collision); R4 (front-fault slow-down) is recorded for every trial too.

Needs a CARLA server (`scripts/carla_server.sh`; wait about 30 s after its start before running this) and a GPU for
SAM 3; the method checks_only needs no VLM.

    uv run python scripts/falsify_carla.py --hazards pedestrian lead_brake --budget 40 --tag v1 \
        --out-root /data/maxshen/recoverdrive/outputs/falsify

--resume reruns a stopped campaign with the same tag: finished trials are read back from their folders, so only the
missing trials run in CARLA. --confirm K --from-tags A B replays the K lowest-R5 scenarios per hazard of those
campaigns with --method (ours_full, which needs the VLM server of configs/e3.yaml).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

from recoverdrive.falsify import TrialResult, campaign, resume_mismatch, select_worst_carla
from recoverdrive.requirements import CARLA_SENSORS, carla_trace, evaluate
from recoverdrive.runner import git_state
from recoverdrive.scenario import CARLA_SEARCH, scenario_from_params

REPO = Path(__file__).resolve().parents[1]
HAZARDS = ["lead_brake", "pedestrian"]  # E3: route i uses HAZARDS[i % 2]


def load_run_e3():
    spec = importlib.util.spec_from_file_location("run_e3", REPO / "scripts" / "run_e3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def finite(x: float) -> float | None:
    return x if math.isfinite(x) else None


def score(tdir: Path, res: dict) -> tuple[dict, object]:
    """R4/R5 of one finished run, from the step log in its folder."""
    steps = [json.loads(line) for f in tdir.glob("*.jsonl") for line in f.read_text().splitlines() if line.strip()]
    trace, _ = carla_trace(steps, res, CARLA_SENSORS)
    specs = evaluate(trace, ("R4", "R5"), CARLA_SENSORS)
    r5, r4 = specs["R5"], specs["R4"]
    fields = {"collision": res.get("collision"), "route_completion": res.get("route_completion"),
              "R5": finite(r5.robustness), "R4": finite(r4.robustness), "satisfied": r5.satisfied and r4.satisfied}
    return fields, r5


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--hazards", nargs="+", default=HAZARDS)
    p.add_argument("--samplers", nargs="+", default=["random", "tpe"])
    p.add_argument("--budget", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="checks_only")
    p.add_argument("--routes", type=int, default=4)
    p.add_argument("--config", default=str(REPO / "configs" / "e3.yaml"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--tm-port", type=int, default=8100)
    p.add_argument("--town", default="Town01_Opt")
    p.add_argument("--cache", default=None, help="override the config's SAM 3 / VLM cache path")
    p.add_argument("--tag", required=True)
    p.add_argument("--out-root", type=Path, default=Path("outputs/falsify"))
    p.add_argument("--resume", action="store_true", help="continue a stopped campaign with the same tag")
    p.add_argument("--confirm", type=int, default=0, help="replay the K lowest-R5 scenarios per hazard of --from-tags")
    p.add_argument("--from-tags", nargs="*", default=[])
    args = p.parse_args()
    if args.confirm and not args.from_tags:
        raise SystemExit("--confirm needs --from-tags")

    e3 = load_run_e3()
    e3.TM_PORT = args.tm_port
    import carla

    from recoverdrive.cache import KVCache
    from recoverdrive.detectors.sam3 import CachedDetector
    from recoverdrive.vlm import VLMClient, healthy_servers

    cfg = e3.load_e3_config(Path(args.config))
    if args.cache:
        cfg["cache"] = args.cache
    out = args.out_root / f"carla_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    old_rows = read_rows(out / "trials.jsonl") if args.resume else []
    (out / "meta.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "config": cfg,
                                               "git": git_state(REPO), "resumed_trials": len(old_rows)}, indent=2))
    cache = KVCache(cfg["cache"])
    s = cfg["sam3"]
    detector = CachedDetector(cache, s["prompts"], s["score_threshold"], s["model_id"])
    vlm = None
    if args.method not in ("B0", "checks_only"):
        servers = healthy_servers(cfg["vlm"]["base_url"])
        if not servers:
            raise SystemExit(f"no healthy VLM server among {cfg['vlm']['base_url']}")
        vlm = VLMClient(servers, cfg["vlm"]["model"], max_tokens=cfg["vlm"]["max_tokens"],
                        timeout_s=cfg["vlm"]["timeout_s"], cache=cache)
    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    world = client.get_world()
    if not world.get_map().name.endswith(args.town):
        world = client.load_world(args.town)
    settings = world.get_settings()
    run_args = SimpleNamespace(dt=0.1, perception_every=5, route_m=100.0, time_limit_s=45.0, cruise_kmh=30.0,
                               slow_kmh=10.0, ttc_s=2.0, record=False)
    settings.synchronous_mode, settings.fixed_delta_seconds = True, run_args.dt
    world.apply_settings(settings)
    e3.clear_world(world)
    routes = e3.find_routes(world.get_map(), args.routes, run_args.route_m)
    try:
        if args.confirm:
            run_confirm(args, out, e3, client, world, cfg, run_args, routes, detector, vlm)
        else:
            run_campaigns(args, out, old_rows, e3, client, world, cfg, run_args, routes, detector, vlm)
    finally:
        settings.synchronous_mode, settings.fixed_delta_seconds = False, None
        world.apply_settings(settings)


def run_campaigns(args, out, old_rows, e3, client, world, cfg, run_args, routes, detector, vlm) -> None:
    def key(r: dict) -> tuple:
        return r["hazard"], r["sampler"], r["trial"]

    old = {key(r): r["params"] for r in old_rows}
    rows: list[dict] = []
    for hazard in args.hazards:
        idx = [i for i in range(len(routes)) if HAZARDS[i % 2] == hazard]
        space = dict(CARLA_SEARCH[hazard])
        space["route"] = ("cat", idx)
        for sampler in args.samplers:
            cdir = out / f"{hazard}_{sampler}"
            counter = iter(range(10**9))

            def objective(raw, hazard=hazard, sampler=sampler, cdir=cdir, counter=counter):
                i = next(counter)
                prev = old.get((hazard, sampler, i))
                if resume_mismatch(prev, raw):
                    raise SystemExit(f"resume mismatch: {hazard} {sampler} trial {i} drew {raw}, earlier {prev}")
                tdir = cdir / f"trial_{i:04d}"
                if prev is None and tdir.exists() and any(tdir.iterdir()):
                    # A trial that was cut off: keep its files aside instead of mixing them with the new run.
                    tdir.rename(tdir.with_name(f"{tdir.name}.partial-{int(time.time())}"))
                tdir.mkdir(parents=True, exist_ok=True)
                sc = scenario_from_params(hazard, raw)
                res = e3.run_route(client, world, cfg, run_args, raw["route"], routes[raw["route"]], hazard,
                                   "falsify", args.method, raw["seed"], detector, vlm, tdir, scenario=sc)
                fields, r5 = score(tdir, res)
                rows.append({"hazard": hazard, "sampler": sampler, "trial": i, "params": raw, "scenario": vars(sc),
                             **fields, "run_dir": str(tdir)})
                done = {key(r) for r in rows}  # keep earlier rows not reached yet, in case this run stops too
                rest = [r for r in old_rows if key(r) not in done]
                (out / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows + rest))
                return TrialResult(r5.robustness, r5.satisfied, r5.n_violations, raw, str(tdir))

            results = campaign(objective, space, sampler, args.budget, args.seed)
            viol = sum(not r.satisfied for r in results)
            print(f"{hazard} {sampler}: {len(results)} runs, {viol} R5 violations", flush=True)
    md = [f"# CARLA falsification ({args.tag})", "", f"Method: {args.method}; budget {args.budget} per campaign.", "",
          "| hazard | sampler | runs | collisions | R5 violations | R4 violations | min R5 robustness (m or rank) |",
          "|---|---|---|---|---|---|---|"]
    for hazard in args.hazards:
        for sampler in args.samplers:
            rs = [r for r in rows if r["hazard"] == hazard and r["sampler"] == sampler]
            r5 = [r["R5"] for r in rs if r["R5"] is not None]
            md.append(f"| {hazard} | {sampler} | {len(rs)} | {sum(bool(r['collision']) for r in rs)} | "
                      f"{sum(r['R5'] is not None and r['R5'] < 0 for r in rs)} | "
                      f"{sum(r['R4'] is not None and r['R4'] < 0 for r in rs)} | {min(r5) if r5 else 'inf'} |")
    md += ["", f"git: {git_state(REPO)}"]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def run_confirm(args, out, e3, client, world, cfg, run_args, routes, detector, vlm) -> None:
    src = [r | {"from_tag": t} for t in args.from_tags for r in read_rows(args.out_root / f"carla_{t}" / "trials.jsonl")]
    picked = select_worst_carla(src, args.confirm)
    print(f"replaying {len(picked)} scenarios of {len(src)} trials with {args.method}", flush=True)
    rows = []
    for r in picked:
        hazard, raw = r["hazard"], r["params"]
        tdir = out / f"{hazard}_confirm" / f"{r['from_tag']}_trial_{r['trial']:04d}"
        tdir.mkdir(parents=True, exist_ok=True)
        res = e3.run_route(client, world, cfg, run_args, raw["route"], routes[raw["route"]], hazard, "falsify",
                           args.method, raw["seed"], detector, vlm, tdir, scenario=scenario_from_params(hazard, raw))
        fields, _ = score(tdir, res)
        rows.append({"hazard": hazard, "from_tag": r["from_tag"], "trial": r["trial"], "params": raw,
                     "checks_only": {k: r[k] for k in ("collision", "R5", "R4", "satisfied")},
                     args.method: fields, "run_dir": str(tdir)})
        (out / "confirm.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    md = [f"# CARLA confirmation ({args.tag})", "",
          (f"The {args.confirm} lowest-R5 scenarios per hazard of {', '.join(args.from_tags)}, replayed with "
           f"{args.method}."), "",
          f"| hazard | source | R5 checks_only | R5 {args.method} | collision checks_only | collision {args.method} |",
          "|---|---|---|---|---|---|"]
    for x in rows:
        c, m = x["checks_only"], x[args.method]
        md.append(f"| {x['hazard']} | {x['from_tag']} #{x['trial']} | {c['R5']} | {m['R5']} | {c['collision']} | "
                  f"{m['collision']} |")
    md += ["", f"git: {git_state(REPO)}"]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
