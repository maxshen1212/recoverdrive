"""E3 closed-loop safety in CARLA 0.9.16 (spec §3.4, §5.3): B0, B1, Ours-full on short routes with hazards.

Setup (plan D11): the ego follows a straight route with a simple lane-keeping controller that ignores other road users;
the vehicle's own brake fires when time-to-collision < 2 s, using objects the front camera confirms (SAM 3 matches),
or LiDAR objects when the monitor has isolated CAM_FRONT. The monitor (method) runs at 2 Hz and its shield action
caps the speed (slow_down) or stops the car (stop). Hazards: a lead vehicle brakes hard, or a pedestrian crosses.
Conditions: clean, or C1 on CAM_FRONT during the hazard.

Start the server first:  ./scripts/carla_server.sh   (then)
    CUDA_VISIBLE_DEVICES=3 uv run python scripts/run_e3.py --methods B0 B1 ours_full --seeds 0 1 2 --tag e3
"""

import argparse
import json
import queue
import time
from pathlib import Path

import carla
import numpy as np
import yaml

from recoverdrive.adapters.carla_adapter import (
    DEFAULT_CAMERAS,
    LIDAR_MOUNT,
    VEHICLE_CLASSES,
    actor_box,
    bgra_to_rgb,
    build_frame,
)
from recoverdrive.cache import KVCache
from recoverdrive.checks import CheckConfig, CheckThresholds, _project, match_boxes, run_checks
from recoverdrive.detectors.sam3 import CachedDetector
from recoverdrive.faults import inject_c1
from recoverdrive.geometry import to_sensor_frame
from recoverdrive.methods import FrameContext, build_methods
from recoverdrive.metrics import markdown_table
from recoverdrive.recording import JsonlWriter
from recoverdrive.render import camera_mosaic, lidar_bev
from recoverdrive.runner import frame_seed, points_in_ego, sensors_of
from recoverdrive.shield import acceptable_actions, ttc_override
from recoverdrive.types import Action, FaultLabel, Pose
from recoverdrive.vlm import VLMClient, healthy_servers, image_part, summarize_checks

IDENTITY = Pose(np.array([1.0, 0, 0, 0]), np.zeros(3))
TM_PORT = 8100  # the TM default (8000) is taken by vLLM


def find_routes(cmap, n: int, length: float) -> list:
    """Spawn points followed by >= length + 40 m of straight, junction-free lane, at least 60 m apart."""
    routes = []
    for sp in sorted(cmap.get_spawn_points(), key=lambda t: (round(t.location.x), round(t.location.y))):
        wp = cmap.get_waypoint(sp.location)
        ok, cur = True, wp
        for _ in range(int((length + 40) / 5)):
            nxt = cur.next(5.0)
            if not nxt or nxt[0].is_junction:
                ok = False
                break
            cur = nxt[0]
            dyaw = (cur.transform.rotation.yaw - wp.transform.rotation.yaw + 180) % 360 - 180
            if abs(dyaw) > 8:
                ok = False
                break
        if ok and all(sp.location.distance(r.location) > 60 for r in routes):
            routes.append(sp)
        if len(routes) == n:
            break
    return routes


def spawn_scenery(world, bp, wp0, rng: np.random.Generator) -> list:
    """Static road users so the LiDAR-camera cross-check has reference objects (plan D14): parked cars in the
    opposite lane from 30 m behind to 140 m ahead, and pedestrians standing on the right sidewalk."""
    actors = []
    cars = [b for b in bp.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4
            and b.id.split(".")[1] in ("audi", "toyota", "nissan", "mercedes", "ford", "tesla", "mini")]
    walkers = bp.filter("walker.pedestrian.*")
    for d in np.arange(-30.0, 140.0, 14.0):
        wp = wp0.previous(-d)[0] if d < 0 else (wp0.next(d)[0] if d > 0 else wp0)
        opp = wp.get_left_lane()
        if opp is None or opp.lane_type != carla.LaneType.Driving or rng.random() < 0.25:
            continue
        tf = opp.transform
        tf.location.z += 0.3
        a = world.try_spawn_actor(cars[int(rng.integers(len(cars)))], tf)
        if a is not None:
            a.set_simulate_physics(False)
            actors.append(a)
    for d in (10.0, 55.0, 95.0, 125.0):
        wp = wp0.next(d)[0]
        right = wp.transform.get_right_vector()
        loc = wp.transform.location + carla.Location(x=right.x * 4.5, y=right.y * 4.5, z=1.0)
        a = world.try_spawn_actor(walkers[int(rng.integers(len(walkers)))], carla.Transform(loc))
        if a is not None:
            actors.append(a)
    return actors


class EgoController:
    """Lane keeping (pursue the lane centre 6 m ahead) and P speed control. Toggling Traffic Manager autopilot on
    and off every step left a stuck throttle in CARLA 0.9.16, so the ego is not driven by TM."""

    def __init__(self, cmap):
        self.cmap = cmap

    def __call__(self, ego, target_kmh: float, brake: float | None = None) -> "carla.VehicleControl":
        tf = ego.get_transform()
        ahead = self.cmap.get_waypoint(tf.location).next(6.0)[0].transform.location
        yaw = np.radians(tf.rotation.yaw)
        err = (np.arctan2(ahead.y - tf.location.y, ahead.x - tf.location.x) - yaw + np.pi) % (2 * np.pi) - np.pi
        steer = float(np.clip(1.2 * err, -0.5, 0.5))  # UE: positive yaw error = target to the right = steer right
        if brake is not None:
            return carla.VehicleControl(throttle=0.0, brake=brake, steer=steer)
        v = ego.get_velocity()
        e = target_kmh / 3.6 - float(np.hypot(v.x, v.y))
        if e > 0.3:
            return carla.VehicleControl(throttle=float(np.clip(0.3 + 0.12 * e, 0.0, 0.75)), steer=steer)
        if e < -1.0:
            return carla.VehicleControl(throttle=0.0, brake=float(np.clip(-0.12 * e, 0.0, 0.5)), steer=steer)
        return carla.VehicleControl(throttle=0.25 if e > -0.3 else 0.0, steer=steer)


class Sensors:
    def __init__(self, world, ego, bp):
        self.actors, self.queues = [], {}
        for spec in DEFAULT_CAMERAS:
            cbp = bp.find("sensor.camera.rgb")
            cbp.set_attribute("image_size_x", str(spec.width))
            cbp.set_attribute("image_size_y", str(spec.height))
            cbp.set_attribute("fov", str(spec.fov))
            tf = carla.Transform(carla.Location(spec.x, spec.y, spec.z), carla.Rotation(yaw=spec.yaw))
            self._add(world.spawn_actor(cbp, tf, attach_to=ego), spec.name)
        lbp = bp.find("sensor.lidar.ray_cast")
        for k, v in {"channels": "32", "range": "70", "rotation_frequency": "10", "points_per_second": "400000",
                     "upper_fov": "10.67", "lower_fov": "-30.67", "dropoff_general_rate": "0.0"}.items():
            lbp.set_attribute(k, v)
        self._add(world.spawn_actor(lbp, carla.Transform(carla.Location(*LIDAR_MOUNT)), attach_to=ego), "LIDAR")
        self.collisions = []
        col = world.spawn_actor(bp.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
        col.listen(lambda e: self.collisions.append(e.other_actor.type_id))
        self.actors.append(col)

    def _add(self, actor, name):
        q = queue.Queue()
        actor.listen(q.put)
        self.actors.append(actor)
        self.queues[name] = q

    def get(self, frame_id: int, timeout: float = 10.0) -> dict:
        out = {}
        for name, q in self.queues.items():
            while True:
                data = q.get(timeout=timeout)
                if data.frame == frame_id:
                    out[name] = data
                    break
        return out

    def destroy(self):
        for a in self.actors:
            if a.is_alive:
                a.stop() if hasattr(a, "stop") else None
                a.destroy()


def gt_boxes_and_velocities(world, ego_id: int):
    boxes, vel = [], {}
    for a in world.get_actors():
        tid = a.type_id
        if a.id == ego_id or not tid.startswith(("vehicle.", "walker.")):
            continue
        if tid.startswith("walker."):
            cat = "pedestrian"
        else:
            cat = VEHICLE_CLASSES.get(a.attributes.get("base_type", "car"), "car")
        bb = a.bounding_box
        boxes.append(actor_box(np.array(a.get_transform().get_matrix()), (bb.location.x, bb.location.y,
                                                                          bb.location.z),
                               (bb.extent.x, bb.extent.y, bb.extent.z), cat, token=str(a.id)))
        v = a.get_velocity()
        vel[str(a.id)] = np.array([v.x, -v.y, v.z])
    return boxes, vel


def ttc_of(frame, boxes, vel, ego_vel_rh) -> float | None:
    """Smallest time-to-collision over objects in the ego lane corridor (|y| < 1.8 m, 0 < x < 45 m)."""
    best = None
    fwd = frame.ego_pose.matrix()[:3, 0]
    v_ego = float(ego_vel_rh @ fwd)
    for b in boxes:
        local = to_sensor_frame(b, frame.ego_pose, IDENTITY)
        x, y = local.center[0], local.center[1]
        if not (0 < x < 45 and abs(y) < 1.8 + local.wlh[0] / 2):
            continue
        gap = x - local.wlh[1] / 2 - 2.4
        closing = v_ego - float(vel.get(b.token, np.zeros(3)) @ fwd)
        if closing > 0.1:
            t = max(gap, 0.0) / closing
            best = t if best is None else min(best, t)
    return best


def camera_confirmed(frame, all_boxes, dets, cfg: CheckConfig):
    """GT objects that a CAM_FRONT detection confirms (the vehicle's camera-based perception)."""
    proj, _, kept = _project(frame.cameras["CAM_FRONT"], all_boxes, cfg.min_box_px)
    det = np.array([d.box for d in dets.get("CAM_FRONT", [])], float).reshape(-1, 4)
    return [all_boxes[kept[g]].box for g, _, _ in match_boxes(proj, det, cfg.iou_threshold)]


def run_route(client, world, cfg, args, route_idx, route_tf, hazard, condition, method_name, seed, detector, vlm,
              out_dir: Path) -> dict:
    name = f"r{route_idx}_{hazard}_{condition}_{method_name}_s{seed}"
    done = out_dir / f"{name}.json"
    if done.exists():
        return json.loads(done.read_text())
    rng = np.random.default_rng([seed, route_idx])
    bp = world.get_blueprint_library()
    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(seed)
    cmap = world.get_map()
    actors = []
    ego = world.spawn_actor(bp.find("vehicle.lincoln.mkz_2020"), route_tf)
    actors.append(ego)
    sensors = Sensors(world, ego, bp)
    drive = EgoController(cmap)
    start = np.array([route_tf.location.x, route_tf.location.y])
    heading = np.array([np.cos(np.radians(route_tf.rotation.yaw)), np.sin(np.radians(route_tf.rotation.yaw))])
    wp0 = cmap.get_waypoint(route_tf.location)
    trigger_m = 40.0 + float(rng.uniform(-3, 3))
    lead = walker = None
    if hazard == "lead_brake":
        tf = wp0.next(25.0)[0].transform
        tf.location.z += 0.5
        lead = world.spawn_actor(bp.find("vehicle.audi.tt"), tf)
        actors.append(lead)
        lead.set_autopilot(True, TM_PORT)
        tm.ignore_vehicles_percentage(lead, 100)
        tm.ignore_lights_percentage(lead, 100)
        tm.ignore_signs_percentage(lead, 100)
        tm.auto_lane_change(lead, False)
        tm.set_desired_speed(lead, args.cruise_kmh)
    else:
        wp = wp0.next(trigger_m + 18.0)[0]  # a blind ego reaches it ~1.9 s later, when the walker is mid-lane
        right = wp.transform.get_right_vector()
        loc = wp.transform.location + carla.Location(x=right.x * 3.0, y=right.y * 3.0, z=1.0)
        walker = world.spawn_actor(bp.filter("walker.pedestrian.0001")[0], carla.Transform(loc))
        actors.append(walker)
        cross_dir = carla.Vector3D(-right.x, -right.y, 0.0)
    actors += spawn_scenery(world, bp, wp0, rng)
    methods = build_methods([method_name], vlm, cfg)
    method = methods[0]
    check_cfg, thr = CheckConfig(**cfg["checks"]), CheckThresholds(**cfg["thresholds"])
    cams = [c.name for c in DEFAULT_CAMERAS]
    writer = JsonlWriter(out_dir / f"{name}.jsonl")
    action = Action("continue", "start")
    ttc_brake = False
    prev_frame = prev_report = prev_mosaic = None
    triggered_at = None
    steps, t0 = 0, time.perf_counter()
    result = {"run": name, "route": route_idx, "hazard": hazard, "condition": condition, "method": method_name,
              "seed": seed, "collision": False, "collision_with": None}
    progress = 0.0
    n_ticks = int(args.time_limit_s / args.dt)
    for tick in range(n_ticks):
        frame_id = world.tick()
        data = sensors.get(frame_id)
        loc = ego.get_location()
        progress = float((np.array([loc.x, loc.y]) - start) @ heading)
        if sensors.collisions:
            result.update(collision=True, collision_with=sensors.collisions[0])
            break
        if progress >= args.route_m:
            break
        # hazard triggers
        if triggered_at is None and progress >= trigger_m:
            triggered_at = tick
        if hazard == "lead_brake" and triggered_at is not None:
            dt_h = (tick - triggered_at) * args.dt
            if dt_h < 3.0:
                lead.set_autopilot(False, TM_PORT)
                lead.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            elif dt_h < 3.0 + args.dt:
                lead.set_autopilot(True, TM_PORT)
        if hazard == "pedestrian" and triggered_at is not None:
            dt_h = (tick - triggered_at) * args.dt
            walker.apply_control(carla.WalkerControl(direction=cross_dir, speed=1.6 if dt_h < 6.0 else 0.0))
        # perception + monitor at 2 Hz
        if tick % args.perception_every == 0:
            images = {c: bgra_to_rgb(data[c].raw_data, data[c].width, data[c].height) for c in cams}
            pts = np.frombuffer(data["LIDAR"].raw_data, dtype=np.float32).reshape(-1, 4)
            boxes, vel = gt_boxes_and_velocities(world, ego.id)
            frame = build_frame(name, steps, int(data["LIDAR"].timestamp * 1e6), np.array(ego.get_transform()
                                                                                          .get_matrix()),
                                images, DEFAULT_CAMERAS, pts, boxes)
            label = FaultLabel.none()
            # C1 covers the hazard stretch of the route (25 m before to 30 m after the trigger point), whatever
            # the ego's speed, so slowing down cannot move the fault away from the hazard.
            if condition == "C1" and trigger_m - 25.0 <= progress <= trigger_m + 30.0:
                frame = inject_c1(frame, "CAM_FRONT", "CAM_BACK")
                label = FaultLabel("C1", "camera:CAM_FRONT", "high", 0)
            dets = {c: detector.detect(frame.cameras[c].image) for c in cams}
            check = run_checks(frame, prev_frame, prev_report, dets, check_cfg, thr, cams)
            summary = summarize_checks(check.report, check_cfg.max_range_m, check_cfg.iou_threshold,
                                       check_cfg.min_points_in_box, cfg["data"].get("nominal_lidar_points"))
            mosaic = image_part(camera_mosaic({c: frame.cameras[c].image for c in cams}, order=cams, cols=2))
            bev = image_part(lidar_bev(points_in_ego(frame)))
            ctx = FrameContext(frame, label, check, summary, mosaic, prev_mosaic, bev, sensors_of(cams),
                               frame_seed(seed, name, steps), None, dets, prev_frame)
            out = method.step(ctx)[method_name]
            isolated = out.recovery.isolated_sensors if out.recovery else []
            perceived = (check.objects if "camera:CAM_FRONT" in isolated
                         else camera_confirmed(frame, check.all_boxes, dets, check_cfg))
            perceived = [o.box if hasattr(o, "box") else o for o in perceived]
            v = ego.get_velocity()
            ttc = ttc_of(frame, perceived, vel, np.array([v.x, -v.y, v.z]))
            action = ttc_override(out.action, ttc, args.ttc_s)
            ttc_brake = action.reason.startswith("TTC")
            writer.write({"step": steps, "tick": tick, "progress_m": round(progress, 2), "label": label,
                          "monitor_action": out.action.action, "action": action.action, "reason": action.reason,
                          "ttc": ttc, "isolated": isolated, "state": out.recovery.final_state if out.recovery else None,
                          "acceptable": acceptable_actions(label), "speed_mps": float(np.linalg.norm([v.x, v.y])),
                          "flags": check.report.flags, "n_calls": out.n_calls})
            prev_frame, prev_report, prev_mosaic = frame, check.report, mosaic
            steps += 1
        # apply the action every tick
        if action.action == "stop":
            ego.apply_control(drive(ego, 0.0, brake=1.0 if ttc_brake else 0.6))
        else:
            ego.apply_control(drive(ego, args.cruise_kmh if action.action == "continue" else args.slow_kmh))
    writer.close()
    sensors.destroy()
    for a in actors:
        if a.is_alive:
            a.destroy()
    world.tick()
    recs = [json.loads(line) for line in (out_dir / f"{name}.jsonl").read_text().splitlines()]
    unnecessary = sum(r["monitor_action"] == "stop" and "stop" not in r["acceptable"] for r in recs)
    result.update(route_completion=min(max(progress, 0.0) / args.route_m, 1.0), steps=len(recs),
                  unnecessary_stop_steps=unnecessary, unnecessary_stop_rate=unnecessary / max(len(recs), 1),
                  stop_steps=sum(r["action"] == "stop" for r in recs), wall_s=round(time.perf_counter() - t0, 1),
                  sim_s=round((tick + 1) * args.dt, 1))
    done.write_text(json.dumps(result))
    return result


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_e3_config(path: Path) -> dict:
    """The frozen E1/E2 config (`base`) with CARLA-specific `overrides` (sensor-dependent values only)."""
    e3 = yaml.safe_load(path.read_text())
    base = yaml.safe_load(Path(e3["base"]).read_text())
    return deep_merge(base, e3.get("overrides", {}))


def e3_table(results: list[dict]) -> str:
    rows = []
    for m in dict.fromkeys(r["method"] for r in results):
        for cond in ("clean", "C1", "all"):
            rs = [r for r in results if r["method"] == m and (cond == "all" or r["condition"] == cond)]
            if not rs:
                continue
            rows.append({"method": m, "condition": cond, "runs": len(rs),
                         "collision_rate": float(np.mean([r["collision"] for r in rs])),
                         "route_completion": float(np.mean([r["route_completion"] for r in rs])),
                         "unnecessary_stops": float(np.mean([r["unnecessary_stop_rate"] for r in rs]))})
    return markdown_table(rows, ["method", "condition", "runs", "collision_rate", "route_completion",
                                 "unnecessary_stops"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/e3.yaml", help="CARLA overrides on top of its `base` config")
    p.add_argument("--methods", nargs="+", default=["B0", "B1", "ours_full"])
    p.add_argument("--conditions", nargs="+", default=["clean", "C1"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--routes", type=int, default=4)
    p.add_argument("--tag", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--town", default="Town01_Opt")  # has 6 straight junction-free routes; Town03 has 1
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--perception-every", type=int, default=5)
    p.add_argument("--route-m", type=float, default=100.0)
    p.add_argument("--time-limit-s", type=float, default=45.0)
    p.add_argument("--cruise-kmh", type=float, default=30.0)
    p.add_argument("--slow-kmh", type=float, default=10.0)
    p.add_argument("--ttc-s", type=float, default=2.0)
    args = p.parse_args()
    cfg = load_e3_config(Path(args.config))
    out_dir = Path("outputs/e3") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps({"args": vars(args), "config": cfg}, indent=2))

    cache = KVCache(cfg["cache"])
    s = cfg["sam3"]
    detector = CachedDetector(cache, s["prompts"], s["score_threshold"], s["model_id"])
    servers = healthy_servers(cfg["vlm"]["base_url"])
    vlm = VLMClient(servers, cfg["vlm"]["model"], max_tokens=cfg["vlm"]["max_tokens"],
                    timeout_s=cfg["vlm"]["timeout_s"], cache=cache) if servers else None
    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    world = client.get_world()
    if not world.get_map().name.endswith(args.town):
        world = client.load_world(args.town)
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, args.dt
    world.apply_settings(settings)
    routes = find_routes(world.get_map(), args.routes, args.route_m)
    print(f"{len(routes)} routes: {[(round(r.location.x), round(r.location.y)) for r in routes]}")
    hazards = ["lead_brake", "pedestrian"]
    results = []
    try:
        for seed in args.seeds:
            for i, tf in enumerate(routes):
                for cond in args.conditions:
                    for m in args.methods:
                        res = run_route(client, world, cfg, args, i, tf, hazards[i % 2], cond, m, seed, detector, vlm,
                                        out_dir)
                        results.append(res)
                        print(json.dumps(res))
    finally:
        settings.synchronous_mode, settings.fixed_delta_seconds = False, None
        world.apply_settings(settings)
    text = f"# E3 ({args.tag}): {len(results)} runs\n\n" + e3_table(results)
    (out_dir / "summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
