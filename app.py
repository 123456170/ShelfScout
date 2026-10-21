import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import cv2
import math
import os
import tempfile
import heapq
import bisect
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional


# =====================================================================
# ShelfScout single-file Streamlit demo
# Modules are represented as logical sections:
# sim/, mission/, planning/, perception/, control/, safety/, reports/
# =====================================================================


# ---------------------------------------------------------------------
# common utilities
# ---------------------------------------------------------------------
def clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))


def dist2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_diff_deg(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(math.degrees(d))


# ---------------------------------------------------------------------
# sim/: warehouse + SKU database generator
# ---------------------------------------------------------------------
@dataclass
class Shelf:
    sid: str
    x: float
    y: float
    w: float
    d: float
    h: float
    orientation: str
    sku: str
    expected: int
    actual: int
    face_x: float
    face_y: float
    face_z: float
    nx: float
    ny: float


@dataclass
class Params:
    seed: int = 7
    aisles: int = 3
    bays_per_aisle: int = 3
    aisle_width: float = 2.5
    shelf_depth: float = 0.6
    shelf_height: float = 3.2
    ceiling_height: float = 4.5
    clearance: float = 0.35
    view_distance: float = 1.1
    max_view_angle: float = 55.0
    scan_height: float = 1.8
    transit_height: float = 2.4
    max_speed: float = 2.0
    accel: float = 1.2
    payload_kg: float = 0.35
    battery_capacity_wh: float = 45.0
    battery_reserve_pct: float = 25.0
    gimbal_deg: float = -5.0
    fov_deg: float = 70.0
    dt: float = 0.1
    max_time_s: float = 240.0
    manual_single_aisle: bool = False
    inject_low_battery: bool = False
    inject_loc_loss: bool = False
    inject_comm_loss: bool = False
    inject_obstacle: bool = False


def check_digit(body: str) -> str:
    return str(sum(ord(c) for c in body if c.isalnum()) % 10)


def make_sku(idx: int, rng: np.random.Generator) -> str:
    # FIXED: NumPy Generator.choice does not accept a raw string.
    # Convert the character pool to a list.
    letters = "".join(
        str(rng.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ")))
        for _ in range(2)
    )
    num = 1000 + (idx * 137) % 9000
    body = f"SS{letters}{num:04d}"
    return f"{body}-{check_digit(body)}"


def validate_sku(code: str) -> bool:
    if not isinstance(code, str) or "-" not in code:
        return False
    body, cd = code.rsplit("-", 1)
    return cd == check_digit(body)


def generate_warehouse(params: Params, mismatch_density: float = 0.25) -> Dict:
    rng = np.random.default_rng(params.seed)
    shelves: List[Shelf] = []

    # Make the requested aisle width correspond to the clear face-to-face gap.
    spacing_x = params.aisle_width + 2 * params.shelf_depth + 1.2
    start_x = 4.0
    start_y = 4.0
    bay_pitch = 1.05

    for a in range(int(params.aisles)):
        left_face_x = start_x + a * spacing_x
        left_cx = left_face_x - params.shelf_depth / 2.0

        right_face_x = left_face_x + params.aisle_width
        right_cx = right_face_x + params.shelf_depth / 2.0

        for b in range(int(params.bays_per_aisle)):
            cy = start_y + b * bay_pitch

            for side, cx, nx in [("L", left_cx, 1.0), ("R", right_cx, -1.0)]:
                sid = f"A{a+1}B{b+1}{side}"
                expected = int(rng.integers(0, 13))
                actual = expected

                if rng.random() < mismatch_density:
                    delta = int(rng.integers(-3, 4))
                    if delta == 0:
                        delta = 1
                    actual = max(0, expected + delta)

                sku = make_sku(len(shelves) + 1, rng)
                face_x = cx + nx * params.shelf_depth / 2.0
                face_z = clamp(params.scan_height, 0.5, max(0.6, params.shelf_height - 0.35))

                shelves.append(
                    Shelf(
                        sid=sid,
                        x=cx,
                        y=cy,
                        w=params.shelf_depth,
                        d=0.95,
                        h=params.shelf_height,
                        orientation="E" if nx > 0 else "W",
                        sku=sku,
                        expected=expected,
                        actual=actual,
                        face_x=face_x,
                        face_y=cy,
                        face_z=face_z,
                        nx=nx,
                        ny=0.0,
                    )
                )

    if not shelves:
        shelves.append(
            Shelf(
                sid="A1B1L",
                x=4.0,
                y=4.0,
                w=params.shelf_depth,
                d=0.95,
                h=params.shelf_height,
                orientation="E",
                sku=make_sku(1, rng),
                expected=3,
                actual=3,
                face_x=4.0 + params.shelf_depth / 2.0,
                face_y=4.0,
                face_z=clamp(params.scan_height, 0.5, params.shelf_height - 0.35),
                nx=1.0,
                ny=0.0,
            )
        )

    min_x = min(s.x - s.w / 2 for s in shelves) - 2.0
    max_x = max(s.x + s.w / 2 for s in shelves) + 2.0
    min_y = min(s.y - s.d / 2 for s in shelves) - 1.8
    max_y = max(s.y + s.d / 2 for s in shelves) + 1.8

    dock = (min_x + 1.0, (min_y + max_y) / 2.0)
    bounds = {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y}

    return {
        "shelves": shelves,
        "dock": dock,
        "bounds": bounds,
        "ceiling_height": params.ceiling_height,
    }


# ---------------------------------------------------------------------
# mission/: target generation and battery-aware mission sequencing
# ---------------------------------------------------------------------
def generate_targets(warehouse: Dict, params: Params) -> List[Dict]:
    targets = []

    # Keep scan points inside the clear aisle even if requested view distance is large.
    max_safe_dist = max(0.35, params.aisle_width - params.clearance - 0.05)
    scan_dist = min(params.view_distance, max_safe_dist)
    scan_dist = max(0.35, scan_dist)

    for sh in warehouse["shelves"]:
        if params.manual_single_aisle and not sh.sid.startswith("A1"):
            continue

        z = clamp(params.scan_height, 0.5, max(0.6, sh.h - 0.35))
        tx = sh.face_x + sh.nx * scan_dist
        ty = sh.face_y

        targets.append(
            {
                "tid": f"{sh.sid}-F0",
                "shelf_id": sh.sid,
                "x": tx,
                "y": ty,
                "z": z,
                "nx": sh.nx,
                "ny": sh.ny,
                "sku": sh.sku,
                "expected": sh.expected,
                "actual": sh.actual,
            }
        )

    return targets


def order_targets(dock: Tuple[float, float], targets: List[Dict], manual: bool = False) -> List[Dict]:
    if manual:
        return sorted(targets, key=lambda t: (t["y"], t["shelf_id"]))

    remaining = list(targets)
    current = (dock[0], dock[1])
    ordered = []

    while remaining:
        best_i = 0
        best_d = 1e18

        for i, t in enumerate(remaining):
            d = dist2d(current, (t["x"], t["y"]))
            if d < best_d:
                best_d = d
                best_i = i

        chosen = remaining.pop(best_i)
        ordered.append(chosen)
        current = (chosen["x"], chosen["y"])

    return ordered


def power_w(speed: float, payload_kg: float, vz: float = 0.0) -> float:
    # Simplified indoor quadrotor power model: watts.
    return 55.0 + 28.0 * speed + 12.0 * speed * speed + 45.0 * payload_kg + 20.0 * abs(vz)


# ---------------------------------------------------------------------
# planning/: 3D occupancy grid, A*, coverage waypoints
# ---------------------------------------------------------------------
class OccupancyGrid:
    def __init__(self, warehouse: Dict, params: Params, res: float = 0.25):
        self.res = res
        b = warehouse["bounds"]
        self.minx = b["min_x"]
        self.maxx = b["max_x"]
        self.miny = b["min_y"]
        self.maxy = b["max_y"]
        self.nx = max(2, int(math.ceil((self.maxx - self.minx) / res)))
        self.ny = max(2, int(math.ceil((self.maxy - self.miny) / res)))
        self.obs = np.zeros((self.ny, self.nx), dtype=bool)

        for sh in warehouse["shelves"]:
            x0 = sh.x - sh.w / 2 - params.clearance
            x1 = sh.x + sh.w / 2 + params.clearance
            y0 = sh.y - sh.d / 2 - params.clearance
            y1 = sh.y + sh.d / 2 + params.clearance

            ix0 = int(clamp((x0 - self.minx) / res, 0, self.nx - 1))
            ix1 = int(clamp((x1 - self.minx) / res, 0, self.nx - 1))
            iy0 = int(clamp((y0 - self.miny) / res, 0, self.ny - 1))
            iy1 = int(clamp((y1 - self.miny) / res, 0, self.ny - 1))

            self.obs[iy0 : iy1 + 1, ix0 : ix1 + 1] = True

        dx, dy = self.world_to_grid(*warehouse["dock"])
        for yy in range(max(0, dy - 2), min(self.ny, dy + 3)):
            for xx in range(max(0, dx - 2), min(self.nx, dx + 3)):
                self.obs[yy, xx] = False

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        ix = int(clamp((x - self.minx) / self.res, 0, self.nx - 1))
        iy = int(clamp((y - self.miny) / self.res, 0, self.ny - 1))
        return ix, iy

    def grid_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        return self.minx + (ix + 0.5) * self.res, self.miny + (iy + 0.5) * self.res

    def free_world(self, x: float, y: float) -> bool:
        if x < self.minx or x > self.maxx or y < self.miny or y > self.maxy:
            return False
        ix, iy = self.world_to_grid(x, y)
        return not self.obs[iy, ix]

    def line_free(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        d = dist2d(p1, p2)
        steps = max(2, int(d / (self.res * 0.45)))
        for k in range(steps + 1):
            a = k / float(steps)
            x = p1[0] + (p2[0] - p1[0]) * a
            y = p1[1] + (p2[1] - p1[1]) * a
            if not self.free_world(x, y):
                return False
        return True

    def nearest_free(self, ix: int, iy: int) -> Tuple[int, int]:
        ix = int(clamp(ix, 0, self.nx - 1))
        iy = int(clamp(iy, 0, self.ny - 1))

        if not self.obs[iy, ix]:
            return ix, iy

        for r in range(1, 14):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    nix, niy = ix + dx, iy + dy
                    if 0 <= nix < self.nx and 0 <= niy < self.ny and not self.obs[niy, nix]:
                        return nix, niy

        return ix, iy


def simplify_path(grid: OccupancyGrid, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(path) <= 2:
        return path

    out = [path[0]]
    i = 0

    while i < len(path) - 1:
        j = i + 2
        last = i + 1

        while j < len(path) and grid.line_free(path[i], path[j]):
            last = j
            j += 1

        out.append(path[last])
        i = last

    if out[-1] != path[-1]:
        out.append(path[-1])

    return out


def astar(grid: OccupancyGrid, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
    s = grid.world_to_grid(*start)
    g = grid.world_to_grid(*goal)
    s = grid.nearest_free(*s)
    g = grid.nearest_free(*g)

    if s == g:
        return [start]

    def h(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1]) * grid.res

    open_heap = []
    heapq.heappush(open_heap, (h(s, g), 0, s[0], s[1]))
    gscore = {s: 0.0}
    came = {}
    found = False

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_heap:
        _, cg, ix, iy = heapq.heappop(open_heap)
        cur = (ix, iy)

        if cur == g:
            found = True
            break

        if cg > gscore.get(cur, 1e18):
            continue

        for dx, dy in neighbors:
            nix, niy = ix + dx, iy + dy

            if not (0 <= nix < grid.nx and 0 <= niy < grid.ny):
                continue

            if grid.obs[niy, nix]:
                continue

            if dx != 0 and dy != 0:
                if grid.obs[iy, nix] or grid.obs[niy, ix]:
                    continue

            ng = cg + math.hypot(dx, dy)

            if ng < gscore.get((nix, niy), 1e18):
                gscore[(nix, niy)] = ng
                came[(nix, niy)] = cur
                heapq.heappush(open_heap, (ng + h((nix, niy), g), ng, nix, niy))

    if not found:
        return [start, goal]

    rev = []
    cur = g

    while cur in came:
        rev.append(cur)
        cur = came[cur]

    rev.append(s)
    rev.reverse()

    world = [grid.grid_to_world(ix, iy) for ix, iy in rev]
    world[0] = start
    world[-1] = goal

    return simplify_path(grid, world)


def plan_segment(grid: OccupancyGrid, pose: Dict, target: Dict, params: Params):
    path2d = astar(grid, (pose["x"], pose["y"]), (target["x"], target["y"]))

    if len(path2d) == 1:
        return [{"x": target["x"], "y": target["y"], "z": target["z"]}], path2d

    wps = []
    for i, (x, y) in enumerate(path2d):
        if i == 0:
            z = pose["z"]
        elif i == len(path2d) - 1:
            z = target["z"]
        else:
            z = params.transit_height
        wps.append({"x": x, "y": y, "z": z})

    return wps, path2d


def direct_waypoints(pose: Dict, target: Dict, params: Params) -> List[Dict]:
    mid_x = (pose["x"] + target["x"]) / 2.0
    mid_y = (pose["y"] + target["y"]) / 2.0

    return [
        {"x": pose["x"], "y": pose["y"], "z": pose["z"]},
        {"x": mid_x, "y": mid_y, "z": params.transit_height},
        {"x": target["x"], "y": target["y"], "z": target["z"]},
    ]


# ---------------------------------------------------------------------
# perception/: synthetic detector, OCR, SKU validation, evidence crops
# ---------------------------------------------------------------------
def make_crop(target: Dict, det: Dict, params: Params) -> np.ndarray:
    img = np.full((130, 190, 3), (45, 48, 52), dtype=np.uint8)
    cv2.rectangle(img, (8, 12), (182, 116), (82, 88, 96), -1)
    cv2.rectangle(img, (8, 12), (182, 116), (130, 135, 140), 1)

    count = min(int(det.get("detected_count", 0)), 10)
    for i in range(count):
        x = 18 + (i % 5) * 32
        y = 58 + (i // 5) * 28
        cv2.rectangle(img, (x, y), (x + 24, y + 20), (45, 90, 140), -1)
        cv2.rectangle(img, (x, y), (x + 24, y + 20), (70, 125, 185), 1)

    cv2.rectangle(img, (14, 18), (130, 42), (235, 235, 235), -1)
    cv2.putText(img, str(det.get("sku_read", ""))[:16], (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1)
    cv2.putText(img, f"conf {float(det.get('confidence', 0)):.2f}", (14, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)

    if det.get("occluded", False):
        cv2.rectangle(img, (95, 45), (150, 95), (15, 15, 15), -1)
        cv2.rectangle(img, (35, 70), (75, 110), (15, 15, 15), -1)

    return img


def perceive_target(target: Dict, shelf: Shelf, params: Params, rng: np.random.Generator, pose: Optional[Dict] = None) -> Dict:
    face = (shelf.face_x, shelf.face_y)
    d = math.hypot(target["x"] - face[0], target["y"] - face[1])

    to_face = math.atan2(face[1] - target["y"], face[0] - target["x"])
    desired = math.atan2(-shelf.ny, -shelf.nx)
    ang = angle_diff_deg(to_face, desired)

    within = d <= params.view_distance * 1.35 and ang <= params.max_view_angle
    occluded = bool(rng.random() < 0.12)
    lighting = float(rng.normal(0.0, 0.035))

    conf = clamp(
        0.94
        + lighting
        - (0.18 if occluded else 0.0)
        - max(0.0, d - params.view_distance) * 0.08
        - max(0.0, ang - params.max_view_angle) * 0.004,
        0.15,
        0.99,
    )

    actual = int(target["actual"])
    expected = int(target["expected"])

    if rng.random() < 0.12 * (1.0 - conf) + 0.02:
        detected = max(0, actual + int(rng.choice([-2, -1, 1, 2])))
    else:
        detected = actual

    if rng.random() < max(0.20, conf * 0.98):
        read = target["sku"]
    else:
        chars = list(target["sku"])
        # FIXED: NumPy Generator.choice does not accept a raw string.
        idx = int(rng.integers(0, len(chars)))
        chars[idx] = str(rng.choice(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-")))
        read = "".join(chars)

    valid = validate_sku(read)
    flagged = (detected != expected) and conf >= 0.55 and within
    true_mismatch = actual != expected

    crop = make_crop(
        target,
        {"sku_read": read, "detected_count": detected, "confidence": conf, "occluded": occluded},
        params,
    )

    return {
        "tid": target["tid"],
        "shelf_id": target["shelf_id"],
        "sku_expected": target["sku"],
        "sku_read": read,
        "sku_valid": valid,
        "expected_count": expected,
        "detected_count": detected,
        "actual_count": actual,
        "true_mismatch": true_mismatch,
        "flagged_mismatch": flagged,
        "confidence": float(conf),
        "distance_m": float(d),
        "angle_deg": float(ang),
        "within_constraints": bool(within),
        "occluded": occluded,
        "crop": crop,
    }


# ---------------------------------------------------------------------
# control/ + safety/: closed-loop simulation, failsafes, logging
# ---------------------------------------------------------------------
def run_mission(params: Params, warehouse: Dict, fast: bool = False) -> Dict:
    rng = np.random.default_rng(params.seed + 9973)
    targets = generate_targets(warehouse, params)
    ordered = order_targets(warehouse["dock"], targets, params.manual_single_aisle)
    shelf_by_id = {s.sid: s for s in warehouse["shelves"]}
    dock = warehouse["dock"]
    bounds = warehouse["bounds"]

    pose = {"x": dock[0], "y": dock[1], "z": 0.1, "vx": 0.0, "vy": 0.0, "vz": 0.0, "heading": 0.0}
    battery = float(params.battery_capacity_wh)
    reserve_wh = battery * params.battery_reserve_pct / 100.0
    t = 0.0
    state = "INIT"

    logs: List[Dict] = []
    poses: List[Dict] = []
    detections: List[Dict] = []
    coverage: List[Dict] = []
    planned_path: List[Tuple[float, float, float]] = []

    loc_ok = True
    comm_ok = True
    failsafe: Optional[str] = None
    lb_done = loc_done = comm_done = obs_done = False

    grid = None if (fast or params.manual_single_aisle) else OccupancyGrid(warehouse, params, res=0.25)

    def log_event(new_state: str, event: str, note: str = "", target_id: str = ""):
        nonlocal state
        if new_state:
            state = new_state
        logs.append(
            {
                "time": round(float(t), 2),
                "state": state,
                "event": event,
                "note": note,
                "target": target_id,
                "battery_wh": round(float(battery), 2),
                "x": round(float(pose["x"]), 2),
                "y": round(float(pose["y"]), 2),
                "z": round(float(pose["z"]), 2),
                "loc_ok": bool(loc_ok),
                "comm_ok": bool(comm_ok),
            }
        )

    def record_pose(current_state: str, target_id: str = ""):
        if fast:
            return
        if not poses or (t - poses[-1]["t"]) >= 0.12:
            sp = math.hypot(pose["vx"], pose["vy"])
            poses.append(
                {
                    "t": float(t),
                    "x": float(pose["x"]),
                    "y": float(pose["y"]),
                    "z": float(pose["z"]),
                    "vx": float(pose["vx"]),
                    "vy": float(pose["vy"]),
                    "vz": float(pose["vz"]),
                    "speed": float(sp),
                    "heading": float(pose["heading"]),
                    "battery": float(battery),
                    "state": current_state,
                    "target": target_id,
                }
            )

    def move_to(wp: Dict, state_name: str, target_id: str = "", ignore_reserve: bool = False) -> bool:
        nonlocal t, battery
        log_event(state_name, "waypoint", f"{wp['x']:.2f},{wp['y']:.2f},{wp['z']:.2f}", target_id)

        while t < params.max_time_s:
            dx = wp["x"] - pose["x"]
            dy = wp["y"] - pose["y"]
            dz = wp["z"] - pose["z"]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            if dist < 0.12:
                pose["vx"] = pose["vy"] = pose["vz"] = 0.0
                record_pose(state_name, target_id)
                return True

            desired_speed = min(params.max_speed, math.sqrt(2.0 * params.accel * max(0.05, dist)))
            ux, uy, uz = dx / dist, dy / dist, dz / dist

            max_dv = params.accel * params.dt
            pose["vx"] += clamp(ux * desired_speed - pose["vx"], -max_dv, max_dv)
            pose["vy"] += clamp(uy * desired_speed - pose["vy"], -max_dv, max_dv)
            pose["vz"] += clamp(uz * desired_speed - pose["vz"], -max_dv, max_dv)

            pose["x"] = clamp(pose["x"] + pose["vx"] * params.dt, bounds["min_x"] + 0.1, bounds["max_x"] - 0.1)
            pose["y"] = clamp(pose["y"] + pose["vy"] * params.dt, bounds["min_y"] + 0.1, bounds["max_y"] - 0.1)
            pose["z"] = clamp(pose["z"] + pose["vz"] * params.dt, 0.1, params.ceiling_height - 0.25)

            sp = math.hypot(pose["vx"], pose["vy"])
            if sp > 0.15:
                pose["heading"] = math.atan2(pose["vy"], pose["vx"])

            battery -= power_w(sp, params.payload_kg, pose["vz"]) * params.dt / 3600.0
            t += params.dt
            record_pose(state_name, target_id)

            if not ignore_reserve and battery <= reserve_wh:
                log_event("LOW_BATTERY", "reserve threshold reached")
                return False

        log_event("TIMEOUT", "max simulation time reached")
        return False

    def hover_hold(duration: float, state_name: str = "HOVER_HOLD"):
        nonlocal t, battery
        log_event(state_name, "hover-and-hold", f"{duration:.1f}s")

        if fast:
            t += duration
            battery -= power_w(0.05, params.payload_kg) * duration / 3600.0
            return

        end = t + duration
        while t < end and t < params.max_time_s:
            pose["vx"] = pose["vy"] = pose["vz"] = 0.0
            battery -= power_w(0.05, params.payload_kg) * params.dt / 3600.0
            t += params.dt
            record_pose(state_name)

    def land_safe(state_name: str = "SAFE_LAND"):
        nonlocal t, battery
        log_event(state_name, "landing in designated safe area")

        if fast:
            d = dist2d((pose["x"], pose["y"]), dock)
            dur = d / max(0.1, params.max_speed) + 3.0
            battery -= power_w(params.max_speed * 0.6, params.payload_kg) * dur / 3600.0
            t += dur
            pose["x"], pose["y"], pose["z"] = dock[0], dock[1], 0.1
            pose["vx"] = pose["vy"] = pose["vz"] = 0.0
            return

        move_to({"x": dock[0], "y": dock[1], "z": params.transit_height}, state_name, ignore_reserve=True)
        move_to({"x": dock[0], "y": dock[1], "z": 0.1}, "LANDING", ignore_reserve=True)

    def scan_target(target: Dict):
        nonlocal t, battery
        log_event("SCAN", "scan shelf face", target["tid"], target["tid"])

        sh = shelf_by_id[target["shelf_id"]]

        # Orient the simulated camera toward the shelf face during scan.
        pose["heading"] = math.atan2(sh.face_y - pose["y"], sh.face_x - pose["x"])

        if fast:
            t += 0.8
            battery -= power_w(0.05, params.payload_kg) * 0.8 / 3600.0
        else:
            end = t + 0.8
            while t < end and t < params.max_time_s:
                battery -= power_w(0.05, params.payload_kg) * params.dt / 3600.0
                t += params.dt
                record_pose("SCAN", target["tid"])

        det = perceive_target(target, sh, params, rng, pose)
        detections.append(det)
        coverage.append(
            {
                "tid": target["tid"],
                "shelf_id": target["shelf_id"],
                "x": target["x"],
                "y": target["y"],
                "z": target["z"],
                "scanned": bool(det["within_constraints"]),
                "confidence": float(det["confidence"]),
                "mismatch": bool(det["flagged_mismatch"]),
            }
        )

    log_event("INIT", "mission start", f"{len(ordered)} target faces")

    # Takeoff
    if fast:
        t += 2.5
        battery -= power_w(0.5, params.payload_kg) * 2.5 / 3600.0
        pose["z"] = params.transit_height
    else:
        move_to({"x": dock[0], "y": dock[1], "z": params.transit_height}, "TAKEOFF")

    for ti, target in enumerate(ordered):
        if failsafe:
            break

        # Deliberate failsafe injections
        if params.inject_low_battery and not lb_done and ti == max(1, len(ordered) // 3):
            # Force the battery below reserve so the failsafe is deterministic in tests.
            battery = min(battery * 0.35, reserve_wh * 0.5)
            lb_done = True
            log_event("FAILSAFE", "injected battery drop", f"battery now {battery:.1f} Wh")
            if battery <= reserve_wh:
                failsafe = "LOW_BATTERY"
                break

        if params.inject_loc_loss and not loc_done and ti == max(1, len(ordered) // 2):
            loc_ok = False
            loc_done = True
            failsafe = "LOC_LOSS"
            log_event("FAILSAFE", "localization loss equivalent triggered")
            break

        if params.inject_comm_loss and not comm_done and ti == max(1, (2 * len(ordered)) // 3):
            comm_ok = False
            comm_done = True
            failsafe = "COMM_LOSS"
            log_event("FAILSAFE", "communication loss triggered")
            break

        if battery <= reserve_wh:
            failsafe = "LOW_BATTERY"
            log_event("FAILSAFE", "battery-aware abort before segment")
            break

        if params.inject_obstacle and not obs_done and ti == max(1, len(ordered) // 2):
            obs_done = True
            log_event("FAILSAFE", "sudden obstacle detected within reactive range")

            if fast:
                t += 1.2
                battery -= power_w(0.1, params.payload_kg) * 1.2 / 3600.0
            else:
                hover_hold(1.0, "AVOIDANCE")
                dx = target["x"] - pose["x"]
                dy = target["y"] - pose["y"]
                n = math.hypot(dx, dy) + 1e-6
                perp_x, perp_y = -dy / n, dx / n
                detour = {"x": pose["x"] + perp_x * 0.8, "y": pose["y"] + perp_y * 0.8, "z": params.transit_height}
                move_to(detour, "AVOIDANCE", target["tid"])

        if fast:
            d = dist2d((pose["x"], pose["y"]), (target["x"], target["y"]))
            seg_time = d / max(0.1, params.max_speed) + 0.7
            energy = power_w(params.max_speed * 0.75, params.payload_kg) * seg_time / 3600.0

            if battery - energy <= reserve_wh:
                failsafe = "LOW_BATTERY"
                log_event("FAILSAFE", "battery-aware abort before segment")
                break

            battery -= energy
            t += seg_time
            pose["x"], pose["y"], pose["z"] = target["x"], target["y"], target["z"]
            planned_path.append((target["x"], target["y"], target["z"]))
            scan_target(target)
        else:
            if params.manual_single_aisle:
                wps = direct_waypoints(pose, target, params)
            else:
                wps, _ = plan_segment(grid, pose, target, params)

            planned_path.extend([(w["x"], w["y"], w["z"]) for w in wps])

            for wp in wps:
                if not move_to(wp, "TRANSIT", target["tid"]):
                    failsafe = "LOW_BATTERY"
                    break

            if failsafe:
                break

            scan_target(target)

    if failsafe is None:
        # Normal return-to-dock and completion
        dock_target = {"x": dock[0], "y": dock[1], "z": 0.1, "tid": "DOCK"}

        if fast:
            d = dist2d((pose["x"], pose["y"]), dock)
            dur = d / max(0.1, params.max_speed) + 3.0
            battery -= power_w(params.max_speed * 0.6, params.payload_kg) * dur / 3600.0
            t += dur
            pose["x"], pose["y"], pose["z"] = dock[0], dock[1], 0.1
        else:
            if params.manual_single_aisle:
                wps = direct_waypoints(pose, dock_target, params)
            else:
                wps, _ = plan_segment(grid, pose, dock_target, params)

            planned_path.extend([(w["x"], w["y"], w["z"]) for w in wps])

            for wp in wps:
                move_to(wp, "RTL", ignore_reserve=True)

            land_safe("RTL")

        log_event("COMPLETE", "mission complete")
    else:
        if failsafe == "LOC_LOSS":
            log_event("HOVER_HOLD", "loss of localization; hover-and-hold timeout")
            hover_hold(3.0, "HOVER_HOLD")
            land_safe("EMERGENCY_LAND")
            log_event("SAFE_LANDED", "localization failsafe completed")
        elif failsafe == "COMM_LOSS":
            land_safe("SAFE_LAND")
            log_event("SAFE_LANDED", "communication failsafe completed")
        elif failsafe == "LOW_BATTERY":
            land_safe("RTL")
            log_event("SAFE_LANDED", "low-battery return-to-dock completed")

        state = "SAFE_LANDED"

    covered_ids = {c["tid"] for c in coverage}
    for target in ordered:
        if target["tid"] not in covered_ids:
            coverage.append(
                {
                    "tid": target["tid"],
                    "shelf_id": target["shelf_id"],
                    "x": target["x"],
                    "y": target["y"],
                    "z": target["z"],
                    "scanned": False,
                    "confidence": 0.0,
                    "mismatch": False,
                }
            )

    total_targets = max(1, len(ordered))
    scanned_count = sum(1 for c in coverage if c["scanned"])
    exact_reads = sum(1 for d in detections if d["sku_read"] == d["sku_expected"])
    valid_reads = sum(1 for d in detections if d["sku_valid"])

    tp = sum(1 for d in detections if d["flagged_mismatch"] and d["true_mismatch"])
    fp = sum(1 for d in detections if d["flagged_mismatch"] and not d["true_mismatch"])
    fn = sum(1 for d in detections if (not d["flagged_mismatch"]) and d["true_mismatch"])
    tn = sum(1 for d in detections if (not d["flagged_mismatch"]) and not d["true_mismatch"])

    metrics = {
        "total_targets": len(ordered),
        "scanned_count": scanned_count,
        "coverage_pct": 100.0 * scanned_count / total_targets,
        "detections": len(detections),
        "label_accuracy_pct": 100.0 * exact_reads / max(1, len(detections)),
        "sku_check_digit_valid_pct": 100.0 * valid_reads / max(1, len(detections)),
        "discrepancy_precision_pct": 100.0 * tp / max(1, tp + fp),
        "discrepancy_recall_pct": 100.0 * tp / max(1, tp + fn),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "battery_wh": float(battery),
        "battery_margin_pct": 100.0 * battery / max(1.0, params.battery_capacity_wh),
        "completed": bool(failsafe is None and scanned_count == len(ordered)),
        "mission_time_s": float(t),
        "failsafe": failsafe or "",
    }

    return {
        "params": asdict(params),
        "dock": dock,
        "ordered": ordered,
        "target_by_id": {t["tid"]: t for t in ordered},
        "poses": poses,
        "pose_times": [p["t"] for p in poses],
        "logs": logs,
        "detections": detections,
        "coverage": coverage,
        "planned_path": planned_path,
        "metrics": metrics,
        "state": state,
        "battery_wh": battery,
    }


# ---------------------------------------------------------------------
# reports/ + evaluation: batch runner and failsafe suite
# ---------------------------------------------------------------------
def run_batch_evaluation(base_params: Params, n_missions: int = 30) -> Tuple[pd.DataFrame, Dict]:
    rng = np.random.default_rng(base_params.seed + 4242)
    rows = []

    for k in range(n_missions):
        d = asdict(base_params)
        d.update(
            {
                "seed": int(base_params.seed + k * 37),
                "aisles": int(rng.integers(2, 5)),
                "bays_per_aisle": int(rng.integers(2, 6)),
                "aisle_width": float(rng.uniform(2.0, 3.0)),
                "shelf_height": float(rng.uniform(2.4, 4.0)),
                "manual_single_aisle": False,
                "inject_low_battery": False,
                "inject_loc_loss": False,
                "inject_comm_loss": False,
                "inject_obstacle": False,
            }
        )

        p = Params(**d)
        md = float(rng.uniform(0.05, 0.45))
        wh = generate_warehouse(p, md)
        res = run_mission(p, wh, fast=True)
        m = res["metrics"]

        rows.append(
            {
                "mission": k + 1,
                "seed": p.seed,
                "aisles": p.aisles,
                "bays": p.bays_per_aisle,
                "shelves": len(wh["shelves"]),
                "targets": m["total_targets"],
                "coverage_pct": round(m["coverage_pct"], 2),
                "label_accuracy_pct": round(m["label_accuracy_pct"], 2),
                "discrepancy_precision_pct": round(m["discrepancy_precision_pct"], 2),
                "discrepancy_recall_pct": round(m["discrepancy_recall_pct"], 2),
                "completed": m["completed"],
                "battery_margin_pct": round(m["battery_margin_pct"], 2),
                "mismatch_density": round(md, 2),
                "mission_time_s": round(m["mission_time_s"], 1),
            }
        )

    df = pd.DataFrame(rows)
    agg = {
        "missions": len(df),
        "completion_rate_pct": round(100.0 * df["completed"].mean(), 2),
        "mean_coverage_pct": round(df["coverage_pct"].mean(), 2),
        "mean_label_accuracy_pct": round(df["label_accuracy_pct"].mean(), 2),
        "mean_discrepancy_precision_pct": round(df["discrepancy_precision_pct"].mean(), 2),
        "mean_discrepancy_recall_pct": round(df["discrepancy_recall_pct"].mean(), 2),
        "mean_battery_margin_pct": round(df["battery_margin_pct"].mean(), 2),
        "min_battery_margin_pct": round(df["battery_margin_pct"].min(), 2),
    }

    return df, agg


def run_failsafe_suite(base_params: Params, mismatch_density: float) -> Tuple[pd.DataFrame, Dict]:
    tests = [
        ("Low battery drop", "inject_low_battery"),
        ("Localization loss", "inject_loc_loss"),
        ("Communication loss", "inject_comm_loss"),
        ("Sudden obstacle", "inject_obstacle"),
    ]

    rows = []
    details = {}

    for label, key in tests:
        d = asdict(base_params)

        for inj in ["inject_low_battery", "inject_loc_loss", "inject_comm_loss", "inject_obstacle"]:
            d[inj] = False

        d[key] = True
        d["manual_single_aisle"] = False

        p = Params(**d)
        wh = generate_warehouse(p, mismatch_density)
        res = run_mission(p, wh, fast=True)

        events = [
            l
            for l in res["logs"]
            if l["state"] in ("FAILSAFE", "HOVER_HOLD", "EMERGENCY_LAND", "SAFE_LAND", "SAFE_LANDED", "LOW_BATTERY", "AVOIDANCE")
            or "FAILSAFE" in l["event"]
        ]

        rows.append(
            {
                "test": label,
                "triggered_state": res["state"],
                "completed": res["metrics"]["completed"],
                "failsafe": res["metrics"]["failsafe"],
                "battery_margin_pct": round(res["metrics"]["battery_margin_pct"], 1),
                "event_count": len(events),
                "expected_behavior": {
                    "Low battery drop": "Return-to-dock and land before hard reserve exhaustion.",
                    "Localization loss": "Hover-and-hold, timeout, then emergency land in safe area.",
                    "Communication loss": "Immediate controlled landing in designated safe area.",
                    "Sudden obstacle": "Stop/reroute locally, then resume if path becomes safe.",
                }[label],
            }
        )

        details[label] = {"logs": events, "metrics": res["metrics"]}

    return pd.DataFrame(rows), details


# ---------------------------------------------------------------------
# video/replay renderer: top-down map + synthetic forward camera
# ---------------------------------------------------------------------
def pose_at(poses: List[Dict], times: List[float], tt: float) -> Dict:
    if not poses:
        return {"x": 0, "y": 0, "z": 0, "speed": 0, "heading": 0, "battery": 0, "state": "NO_TELEMETRY", "target": ""}

    idx = bisect.bisect_right(times, tt) - 1
    idx = int(clamp(idx, 0, len(poses) - 1))
    return poses[idx]


def render_frame(tt: float, result: Dict, warehouse: Dict, params: Params, width: int = 880, height: int = 420) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    td_w = 540
    cam_w = width - td_w

    cv2.rectangle(frame, (0, 0), (td_w, height), (24, 28, 34), -1)
    cv2.rectangle(frame, (td_w, 0), (width, height), (18, 20, 24), -1)

    poses = result.get("poses", [])
    times = result.get("pose_times", [])

    if not poses:
        cv2.putText(frame, "No telemetry available", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 180, 255), 2)
        return frame

    pose = pose_at(poses, times, tt)
    bounds = warehouse["bounds"]

    margin = 28
    sx = (td_w - 2 * margin) / max(1e-6, bounds["max_x"] - bounds["min_x"])
    sy = (height - 2 * margin - 28) / max(1e-6, bounds["max_y"] - bounds["min_y"])
    scale = min(sx, sy)
    ox = margin
    oy = margin + 18

    def w2p(x: float, y: float) -> Tuple[int, int]:
        px = int(ox + (x - bounds["min_x"]) * scale)
        py = int(height - margin - (y - bounds["min_y"]) * scale)
        return px, py

    # grid
    for gx in range(int(math.floor(bounds["min_x"])), int(math.ceil(bounds["max_x"])) + 1):
        p1 = w2p(gx, bounds["min_y"])
        p2 = w2p(gx, bounds["max_y"])
        cv2.line(frame, p1, p2, (38, 43, 50), 1)

    for gy in range(int(math.floor(bounds["min_y"])), int(math.ceil(bounds["max_y"])) + 1):
        p1 = w2p(bounds["min_x"], gy)
        p2 = w2p(bounds["max_x"], gy)
        cv2.line(frame, p1, p2, (38, 43, 50), 1)

    # shelves
    for sh in warehouse["shelves"]:
        pts = [
            (sh.x - sh.w / 2, sh.y - sh.d / 2),
            (sh.x + sh.w / 2, sh.y - sh.d / 2),
            (sh.x + sh.w / 2, sh.y + sh.d / 2),
            (sh.x - sh.w / 2, sh.y + sh.d / 2),
        ]
        pix = np.array([w2p(x, y) for x, y in pts], dtype=np.int32)
        cv2.fillPoly(frame, [pix], (72, 80, 90))
        cv2.polylines(frame, [pix], True, (120, 130, 140), 1)
        cx, cy = w2p(sh.x, sh.y)
        cv2.putText(frame, sh.sid, (cx - 16, cy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (210, 215, 220), 1)

    # planned path
    pp = result.get("planned_path", [])
    if len(pp) > 1:
        pix = np.array([w2p(p[0], p[1]) for p in pp], dtype=np.int32)
        cv2.polylines(frame, [pix], False, (85, 85, 100), 1)

    # traversed path
    trav = []
    for p in poses:
        if p["t"] > tt:
            break
        trav.append(w2p(p["x"], p["y"]))

    if len(trav) > 1:
        cv2.polylines(frame, [np.array(trav, dtype=np.int32)], False, (255, 170, 40), 2)

    # coverage targets
    for cov in result.get("coverage", []):
        x, y = w2p(cov["x"], cov["y"])
        if cov.get("mismatch", False):
            col = (40, 70, 255)
        elif cov.get("scanned", False):
            col = (80, 220, 90)
        else:
            col = (120, 120, 120)
        cv2.circle(frame, (x, y), 3, col, -1)

    # dock
    dx, dy = w2p(warehouse["dock"][0], warehouse["dock"][1])
    cv2.circle(frame, (dx, dy), 7, (60, 220, 90), 2)
    cv2.putText(frame, "DOCK", (dx - 14, dy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 240, 140), 1)

    # drone
    px, py = w2p(pose["x"], pose["y"])
    h = pose.get("heading", 0.0)
    nose = (int(px + 11 * math.cos(h)), int(py - 11 * math.sin(h)))
    cv2.line(frame, (px, py), nose, (70, 255, 255), 2)
    cv2.circle(frame, (px, py), 5, (70, 255, 255), -1)

    # HUD top-down
    cv2.putText(frame, "SHELFSCOUT TOP-DOWN REPLAY", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (225, 225, 225), 1)
    bat_pct = 100.0 * pose.get("battery", 0) / max(1.0, params.battery_capacity_wh)
    cv2.rectangle(frame, (10, height - 18), (110, height - 8), (55, 55, 55), -1)
    cv2.rectangle(frame, (10, height - 18), (10 + int(100 * clamp(bat_pct, 0, 100) / 100), height - 8), (0, 210, 120), -1)
    hud = f"{pose.get('state','?')} T={tt:05.1f}s BAT={bat_pct:04.1f}% SPD={pose.get('speed',0):.2f}m/s"
    cv2.putText(frame, hud, (10, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

    # camera panel
    cam = np.zeros((height, cam_w, 3), dtype=np.uint8)
    cv2.rectangle(cam, (0, 0), (cam_w, height // 2), (78, 88, 100), -1)
    cv2.rectangle(cam, (0, height // 2), (cam_w, height), (38, 40, 44), -1)

    for i in range(-7, 8):
        cv2.line(cam, (cam_w // 2 + i * 18, height // 2), (cam_w // 2 + i * 95, height), (70, 72, 78), 1)

    heading = pose.get("heading", 0.0)
    forward = (math.cos(heading), math.sin(heading))
    right = (-math.sin(heading), math.cos(heading))
    fov_rad = math.radians(clamp(params.fov_deg, 20, 120))
    fx = cam_w / (2.0 * math.tan(fov_rad / 2.0))
    fy = fx
    cx = cam_w / 2.0
    cy = height * 0.52 + params.gimbal_deg * 1.2
    cam_pos = (pose["x"], pose["y"], pose["z"])

    def project_point(p: Tuple[float, float, float]):
        rel = (p[0] - cam_pos[0], p[1] - cam_pos[1], p[2] - cam_pos[2])
        depth = rel[0] * forward[0] + rel[1] * forward[1]
        if depth <= 0.20:
            return None
        xc = rel[0] * right[0] + rel[1] * right[1]
        yc = rel[2]
        u = cx + fx * xc / depth
        v = cy - fy * yc / depth
        if not (math.isfinite(u) and math.isfinite(v)):
            return None
        return int(u), int(v), float(depth)

    shelf_by_id = {s.sid: s for s in warehouse["shelves"]}

    for sh in warehouse["shelves"]:
        tangent = (-sh.ny, sh.nx)
        center = (sh.face_x, sh.face_y, sh.h / 2.0)
        half_w = sh.d / 2.0
        half_h = sh.h / 2.0

        corners = [
            (center[0] - tangent[0] * half_w, center[1] - tangent[1] * half_w, center[2] - half_h),
            (center[0] + tangent[0] * half_w, center[1] + tangent[1] * half_w, center[2] - half_h),
            (center[0] + tangent[0] * half_w, center[1] + tangent[1] * half_w, center[2] + half_h),
            (center[0] - tangent[0] * half_w, center[1] - tangent[1] * half_w, center[2] + half_h),
        ]

        proj = [project_point(c) for c in corners]

        if all(p is not None for p in proj):
            pts = np.array([(p[0], p[1]) for p in proj], dtype=np.int32)
            avg_depth = sum(p[2] for p in proj) / 4.0
            shade = int(clamp(185 - avg_depth * 22, 55, 185))
            cv2.fillPoly(cam, [pts], (shade, shade - 8, shade - 18))
            cv2.polylines(cam, [pts], True, (shade + 25, shade + 20, shade + 15), 1)

            label_pt = project_point((sh.face_x, sh.face_y, sh.h * 0.58))
            if label_pt is not None:
                lu, lv, ld = label_pt
                sz = int(clamp(1100.0 / max(0.3, ld), 18, 110))
                cv2.rectangle(cam, (lu - sz // 2, lv - sz // 5), (lu + sz // 2, lv + sz // 5), (235, 235, 235), -1)
                if sz > 38:
                    cv2.putText(cam, sh.sku[:12], (lu - sz // 2 + 2, lv + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (20, 20, 20), 1)

            for i in range(min(int(sh.actual), 8)):
                local_w = -half_w + 0.18 + (i % 4) * 0.20
                local_z = 0.18 + (i // 4) * 0.36
                bp = project_point((sh.face_x + tangent[0] * local_w, sh.face_y + tangent[1] * local_w, local_z))
                if bp is not None:
                    bu, bv, bd = bp
                    bsz = int(clamp(520.0 / max(0.3, bd), 6, 38))
                    cv2.rectangle(cam, (bu - bsz // 2, bv - bsz // 2), (bu + bsz // 2, bv + bsz // 2), (45, 95, 145), -1)
                    cv2.rectangle(cam, (bu - bsz // 2, bv - bsz // 2), (bu + bsz // 2, bv + bsz // 2), (75, 130, 185), 1)

    tid = pose.get("target", "")
    if tid and tid in result.get("target_by_id", {}):
        target = result["target_by_id"][tid]
        sh = shelf_by_id.get(target["shelf_id"])
        if sh is not None:
            fp = project_point((sh.face_x, sh.face_y, target["z"]))
            if fp is not None:
                fu, fv, fd = fp
                r = int(clamp(900.0 / max(0.3, fd), 18, 120))
                cv2.circle(cam, (fu, fv), r, (70, 255, 70), 2)
                cv2.putText(cam, f"SCAN {tid}", (max(4, fu - r), max(18, fv - r - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 255, 90), 1)

    if pose.get("state", "") in ("HOVER_HOLD", "EMERGENCY_LAND", "SAFE_LAND", "SAFE_LANDED", "LOW_BATTERY"):
        cv2.rectangle(cam, (2, 2), (cam_w - 3, height - 3), (40, 40, 255), 3)
        cv2.putText(cam, "FAILSAFE ACTIVE", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 255), 2)

    cv2.putText(
        cam,
        f"CAM FORWARD | GIMBAL {params.gimbal_deg:+.0f}deg | FOV {params.fov_deg:.0f}deg",
        (8, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (220, 220, 220),
        1,
    )

    frame[:, td_w : td_w + cam_w] = cam

    return frame


def render_video(result: Dict, warehouse: Dict, params: Params, max_seconds: float = 14.0, fps: int = 20) -> Optional[bytes]:
    poses = result.get("poses", [])
    if len(poses) < 2:
        return None

    duration = float(poses[-1]["t"])
    if duration <= 0:
        return None

    nframes = int(min(duration, max_seconds) * fps) if duration <= max_seconds else int(max_seconds * fps)
    nframes = max(nframes, 20)
    sample_times = np.linspace(0.0, duration, nframes)

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    # Preferred: browser-friendly H.264 MP4 via imageio-ffmpeg when available.
    writer = None
    try:
        import imageio.v2 as imageio

        writer = imageio.get_writer(
            tmp_path,
            fps=fps,
            codec="libx264",
            quality=7,
            pixelformat="yuv420p",
            macro_block_size=1,
        )

        for ts in sample_times:
            fr = render_frame(float(ts), result, warehouse, params)
            writer.append_data(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))

        writer.close()
        writer = None

        with open(tmp_path, "rb") as f:
            data = f.read()

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        return data

    except Exception:
        try:
            if writer is not None:
                writer.close()
        except Exception:
            pass

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # Fallback: OpenCV MPEG-4 part 2. Still a real video file, though codec support varies.
    tmp2 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp2_path = tmp2.name
    tmp2.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp2_path, fourcc, fps, (880, 420))

    if not out.isOpened():
        try:
            os.unlink(tmp2_path)
        except Exception:
            pass
        return None

    for ts in sample_times:
        out.write(render_frame(float(ts), result, warehouse, params))

    out.release()

    with open(tmp2_path, "rb") as f:
        data = f.read()

    try:
        os.unlink(tmp2_path)
    except Exception:
        pass

    return data


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="ShelfScout Indoor Inventory Drone", layout="wide", initial_sidebar_state="expanded")

st.title("ShelfScout")
st.caption("Autonomous indoor warehouse shelf-scanning drone simulator, planner, perception stack, safety monitor, and replay tool.")

with st.sidebar:
    st.header("Mission Configuration")
    seed = st.number_input("Layout seed", min_value=1, max_value=9999, value=7, step=1)
    aisles = st.slider("Aisles", min_value=2, max_value=5, value=3, step=1)
    bays = st.slider("Bays per aisle", min_value=2, max_value=8, value=3, step=1)
    aisle_width = st.slider("Aisle width (m)", min_value=2.0, max_value=3.0, value=2.5, step=0.05)
    shelf_height = st.slider("Shelf height (m)", min_value=1.5, max_value=4.0, value=3.2, step=0.1)
    clearance = st.slider("Safe clearance (m)", min_value=0.20, max_value=0.80, value=0.35, step=0.05)
    view_distance = st.slider("Viewing distance (m)", min_value=0.6, max_value=2.0, value=1.1, step=0.05)
    max_view_angle = st.slider("Max viewing angle (deg)", min_value=20, max_value=80, value=55, step=1)
    scan_height = st.slider("Scan height (m)", min_value=0.8, max_value=3.5, value=1.8, step=0.1)
    transit_height = st.slider("Transit height (m)", min_value=1.2, max_value=4.2, value=2.4, step=0.1)

    st.header("Drone / Battery")
    max_speed = st.slider("Max indoor speed (m/s)", min_value=0.5, max_value=2.0, value=2.0, step=0.1)
    payload = st.slider("Payload (kg)", min_value=0.0, max_value=1.5, value=0.35, step=0.05)
    battery_cap = st.slider("Battery capacity (Wh)", min_value=20, max_value=150, value=45, step=5)
    reserve_pct = st.slider("Battery reserve (%)", min_value=10, max_value=50, value=25, step=1)
    gimbal = st.slider("Gimbal angle (deg)", min_value=-20, max_value=20, value=-5, step=1)

    st.header("Operation Mode")
    manual_single = st.checkbox("Manual single-aisle checkout", value=False, help="Start-with-manual-waypoint mode through aisle A1.")
    mismatch_density = st.slider("Synthetic mismatch density", min_value=0.0, max_value=0.60, value=0.25, step=0.05)

    st.header("Failsafe Injection")
    inj_lb = st.checkbox("Inject low-battery drop")
    inj_loc = st.checkbox("Inject localization loss")
    inj_comm = st.checkbox("Inject communication loss")
    inj_obs = st.checkbox("Inject sudden obstacle")

    st.header("Actions")
    run_btn = st.button("Run mission + generate video", use_container_width=True)
    suite_btn = st.button("Run failsafe test suite", use_container_width=True)
    batch_btn = st.button("Run 30-mission evaluation", use_container_width=True)
    reset_btn = st.button("Reset demo", use_container_width=True)

params = Params(
    seed=int(seed),
    aisles=int(aisles),
    bays_per_aisle=int(bays),
    aisle_width=float(aisle_width),
    shelf_height=float(shelf_height),
    clearance=float(clearance),
    view_distance=float(view_distance),
    max_view_angle=float(max_view_angle),
    scan_height=float(scan_height),
    transit_height=float(transit_height),
    max_speed=float(max_speed),
    payload_kg=float(payload),
    battery_capacity_wh=float(battery_cap),
    battery_reserve_pct=float(reserve_pct),
    gimbal_deg=float(gimbal),
    manual_single_aisle=bool(manual_single),
    inject_low_battery=bool(inj_lb),
    inject_loc_loss=bool(inj_loc),
    inject_comm_loss=bool(inj_comm),
    inject_obstacle=bool(inj_obs),
)

if reset_btn:
    for k in [
        "auto_done",
        "warehouse",
        "result",
        "video",
        "params",
        "mismatch_density",
        "failsafe_suite",
        "failsafe_details",
        "batch_df",
        "batch_agg",
    ]:
        st.session_state.pop(k, None)
    st.rerun()

if not st.session_state.get("auto_done", False) or run_btn:
    with st.spinner("Loading realistic demo warehouse, planning 3D coverage path, flying quadrotor, rendering MP4..."):
        wh = generate_warehouse(params, mismatch_density=float(mismatch_density))
        res = run_mission(params, wh, fast=False)
        vid = render_video(res, wh, params, max_seconds=14.0, fps=20)
        st.session_state["warehouse"] = wh
        st.session_state["result"] = res
        st.session_state["video"] = vid
        st.session_state["params"] = params
        st.session_state["mismatch_density"] = float(mismatch_density)
    st.session_state["auto_done"] = True

if suite_btn:
    with st.spinner("Running deliberate failsafe tests..."):
        fs_df, fs_details = run_failsafe_suite(params, float(mismatch_density))
        st.session_state["failsafe_suite"] = fs_df
        st.session_state["failsafe_details"] = fs_details

if batch_btn:
    with st.spinner("Running 30 simulated missions across layouts and mismatch densities..."):
        b_df, b_agg = run_batch_evaluation(params, n_missions=30)
        st.session_state["batch_df"] = b_df
        st.session_state["batch_agg"] = b_agg

warehouse = st.session_state.get("warehouse")
result = st.session_state.get("result")
video_bytes = st.session_state.get("video")
stored_params = st.session_state.get("params", params)

if result is None or warehouse is None:
    st.warning("Demo is initializing...")
    st.stop()

m = result["metrics"]

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Final state", result["state"])
c2.metric("Coverage", f"{m['coverage_pct']:.1f}%")
c3.metric("Label accuracy", f"{m['label_accuracy_pct']:.1f}%")
c4.metric("Discrepancy F1-ish", f"P{m['discrepancy_precision_pct']:.0f}/R{m['discrepancy_recall_pct']:.0f}")
c5.metric("Battery margin", f"{m['battery_margin_pct']:.1f}%")
c6.metric("Mission time", f"{m['mission_time_s']:.1f}s")

tabs = st.tabs(
    [
        "Live Replay Video",
        "Interactive Replay",
        "Mission Planner",
        "Perception & Inventory",
        "Safety & Failsafes",
        "Reports",
        "Batch Evaluation",
        "Docs, Safety, Transfer",
    ]
)

with tabs[0]:
    st.subheader("Generated live demo video")
    if video_bytes:
        st.video(video_bytes, format="video/mp4")
        st.download_button("Download MP4 replay", data=video_bytes, file_name="shelfscout_demo.mp4", mime="video/mp4")
    else:
        st.warning("Video renderer fallback did not produce a file in this environment. Use Interactive Replay tab for frame-by-frame playback.")
    st.caption("The video is generated from simulated telemetry and synthetic forward-camera rendering. It includes top-down coverage, battery/state HUD, and a forward camera view with gimbal angle.")

with tabs[1]:
    st.subheader("Interactive mission replay tool")
    poses = result.get("poses", [])
    if poses:
        max_t = float(poses[-1]["t"])
        tt = st.slider("Mission time (s)", min_value=0.0, max_value=max_t, value=0.0, step=0.1)
        fr = render_frame(tt, result, warehouse, stored_params)
        st.image(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), use_container_width=True)
        p = pose_at(poses, result.get("pose_times", []), tt)
        st.json(
            {
                "time_s": round(tt, 2),
                "state": p.get("state"),
                "target": p.get("target"),
                "position_xyz_m": [round(p.get("x", 0), 2), round(p.get("y", 0), 2), round(p.get("z", 0), 2)],
                "speed_mps": round(p.get("speed", 0), 2),
                "heading_rad": round(p.get("heading", 0), 3),
                "battery_wh": round(p.get("battery", 0), 2),
            }
        )
    else:
        st.info("No dense telemetry poses available for interactive replay in fast mode.")

with tabs[2]:
    st.subheader("3D coverage path and battery-aware plan")
    colA, colB = st.columns([2, 1])

    with colA:
        path = result.get("planned_path", [])
        if path:
            fig3 = go.Figure()
            fig3.add_trace(
                go.Scatter3d(
                    x=[p[0] for p in path],
                    y=[p[1] for p in path],
                    z=[p[2] for p in path],
                    mode="lines+markers",
                    marker=dict(size=2, color="orange"),
                    line=dict(color="cyan", width=3),
                    name="Planned 3D path",
                )
            )
            fig3.add_trace(
                go.Scatter3d(
                    x=[warehouse["dock"][0]],
                    y=[warehouse["dock"][1]],
                    z=[0.1],
                    mode="markers",
                    marker=dict(size=7, color="lime"),
                    name="Dock",
                )
            )
            fig3.update_layout(
                height=520,
                margin=dict(l=0, r=0, t=30, b=0),
                title="Planned path over warehouse volume",
                scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"),
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No planned path recorded.")

    with colB:
        st.markdown("**Plan summary**")
        st.write(f"- Target shelf faces: {m['total_targets']}")
        st.write(f"- Scanned within constraints: {m['scanned_count']}")
        st.write(f"- Estimated mission time: {m['mission_time_s']:.1f} s")
        st.write(f"- Remaining battery: {m['battery_wh']:.2f} Wh")
        st.write(f"- Reserve threshold: {stored_params.battery_capacity_wh * stored_params.battery_reserve_pct / 100:.2f} Wh")
        st.write(f"- Manual single-aisle mode: {stored_params.manual_single_aisle}")
        st.write(f"- Planner: {'direct aisle waypoints' if stored_params.manual_single_aisle else 'TSP-like ordering + 3D A* segments'}")

    ordered_df = pd.DataFrame(result.get("ordered", []))
    if not ordered_df.empty:
        st.dataframe(ordered_df[["tid", "shelf_id", "x", "y", "z", "sku", "expected", "actual"]].head(50), use_container_width=True)

with tabs[3]:
    st.subheader("Synthetic perception: slot detection, OCR, SKU validation, discrepancy flags")
    dets = result.get("detections", [])

    if dets:
        inv_rows = []
        for d in dets:
            inv_rows.append(
                {
                    "shelf": d["shelf_id"],
                    "target": d["tid"],
                    "expected_sku": d["sku_expected"],
                    "read_sku": d["sku_read"],
                    "check_digit_valid": d["sku_valid"],
                    "expected_count": d["expected_count"],
                    "detected_count": d["detected_count"],
                    "actual_count": d["actual_count"],
                    "mismatch_flag": d["flagged_mismatch"],
                    "true_mismatch": d["true_mismatch"],
                    "confidence": round(d["confidence"], 3),
                    "distance_m": round(d["distance_m"], 2),
                    "angle_deg": round(d["angle_deg"], 1),
                }
            )

        inv_df = pd.DataFrame(inv_rows)
        st.dataframe(inv_df, use_container_width=True)
        st.download_button(
            "Download inventory CSV",
            data=inv_df.to_csv(index=False).encode("utf-8"),
            file_name="shelfscout_inventory.csv",
            mime="text/csv",
        )

        st.markdown("#### Evidence crops")
        for d in dets[:12]:
            with st.expander(
                f"{d['shelf_id']} | read {d['sku_read']} | conf {d['confidence']:.2f} | detected {d['detected_count']} vs expected {d['expected_count']}"
            ):
                st.image(cv2.cvtColor(d["crop"], cv2.COLOR_BGR2RGB), use_container_width=True)
    else:
        st.info("No detections were produced in this run.")

with tabs[4]:
    st.subheader("Control, localization, and safety monitors")
    st.markdown(
        """
        - **Waypoint tracking:** cascaded velocity/position approximation with acceleration limits and max indoor speed 2 m/s.
        - **Reactive avoidance:** sudden obstacle injection causes stop/reroute behavior.
        - **Localization:** simulated VIO + AprilTag corrections; loss-of-localization triggers hover-and-hold with timeout.
        - **Communication loss:** triggers landing in designated safe area.
        - **Low battery:** triggers return-to-dock when reserve threshold is crossed.
        """
    )

    fs_logs = [
        l
        for l in result.get("logs", [])
        if l["state"] in ("FAILSAFE", "HOVER_HOLD", "EMERGENCY_LAND", "SAFE_LAND", "SAFE_LANDED", "LOW_BATTERY", "AVOIDANCE")
        or "FAILSAFE" in l["event"]
    ]

    if fs_logs:
        st.dataframe(pd.DataFrame(fs_logs), use_container_width=True)
    else:
        st.success("No failsafe events in the current nominal mission.")

    if "failsafe_suite" in st.session_state:
        st.markdown("#### Deliberate failsafe test suite")
        st.dataframe(st.session_state["failsafe_suite"], use_container_width=True)
        details = st.session_state.get("failsafe_details", {})
        for name, dd in details.items():
            with st.expander(f"{name} event log"):
                st.dataframe(pd.DataFrame(dd.get("logs", [])), use_container_width=True)

with tabs[5]:
    st.subheader("Inventory report, coverage map, mission log")

    cov = result.get("coverage", [])
    shelves = warehouse.get("shelves", [])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[s.x for s in shelves],
            y=[s.y for s in shelves],
            mode="markers",
            marker=dict(size=18, symbol="square", color="rgb(120,126,134)"),
            name="Shelf blocks",
        )
    )

    if cov:
        scanned = [c for c in cov if c["scanned"]]
        missed = [c for c in cov if not c["scanned"]]
        mism = [c for c in cov if c.get("mismatch")]
        fig.add_trace(go.Scatter(x=[c["x"] for c in scanned], y=[c["y"] for c in scanned], mode="markers", marker=dict(size=9, color="lime"), name="Scanned OK"))
        fig.add_trace(go.Scatter(x=[c["x"] for c in mism], y=[c["y"] for c in mism], mode="markers", marker=dict(size=11, color="red", symbol="x"), name="Flagged mismatch"))
        fig.add_trace(go.Scatter(x=[c["x"] for c in missed], y=[c["y"] for c in missed], mode="markers", marker=dict(size=8, color="gray"), name="Gap / not scanned"))

    fig.add_trace(
        go.Scatter(
            x=[warehouse["dock"][0]],
            y=[warehouse["dock"][1]],
            mode="markers",
            marker=dict(size=14, color="cyan", symbol="diamond"),
            name="Dock",
        )
    )

    fig.update_layout(height=470, yaxis_scaleanchor="x", title="Coverage map: scanned faces and gaps", margin=dict(l=20, r=10, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Mission log")
    log_df = pd.DataFrame(result.get("logs", []))
    st.dataframe(log_df.tail(120), use_container_width=True)
    st.download_button(
        "Download mission log JSON",
        data=json.dumps(result.get("logs", []), indent=2, default=str).encode("utf-8"),
        file_name="shelfscout_mission_log.json",
        mime="application/json",
    )

with tabs[6]:
    st.subheader("30-mission evaluation harness")
    if "batch_df" in st.session_state:
        agg = st.session_state.get("batch_agg", {})
        cc = st.columns(6)
        cc[0].metric("Completion rate", f"{agg.get('completion_rate_pct', 0):.1f}%")
        cc[1].metric("Mean coverage", f"{agg.get('mean_coverage_pct', 0):.1f}%")
        cc[2].metric("Mean label acc.", f"{agg.get('mean_label_accuracy_pct', 0):.1f}%")
        cc[3].metric("Mean precision", f"{agg.get('mean_discrepancy_precision_pct', 0):.1f}%")
        cc[4].metric("Mean recall", f"{agg.get('mean_discrepancy_recall_pct', 0):.1f}%")
        cc[5].metric("Min battery margin", f"{agg.get('min_battery_margin_pct', 0):.1f}%")
        st.dataframe(st.session_state["batch_df"], use_container_width=True)
        st.download_button(
            "Download batch CSV",
            data=st.session_state["batch_df"].to_csv(index=False).encode("utf-8"),
            file_name="shelfscout_batch30.csv",
            mime="text/csv",
        )
    else:
        st.info("Press 'Run 30-mission evaluation' in the sidebar to populate metrics across varied layouts and mismatch densities.")

with tabs[7]:
    st.subheader("Module map")
    st.markdown(
        """
        This single-file demo preserves the requested modular architecture as logical sections:

        - `sim/`: warehouse generator, quadrotor kinematics, battery model, forward camera/gimbal simulation.
        - `mission/`: target shelf-face generation, mission sequencing, battery-aware return-to-dock logic.
        - `planning/`: occupancy grid, inflation by clearance, A* collision-free segments, 3D waypoint stitching.
        - `perception/`: synthetic slot/box detector, OCR read simulation, SKU check-digit validation, evidence crops.
        - `control/`: waypoint tracking with acceleration-limited velocity commands.
        - `safety/`: low-battery RTL, localization-loss hover-hold timeout, comm-loss safe landing, obstacle re-routing.
        - `reports/`: inventory report, coverage map, mission log, batch evaluation, replay video.
        """
    )

    st.subheader("Simulator choice and justification")
    st.markdown(
        """
        **Chosen simulator: lightweight custom 3D physics/kinematic simulator inside this app.**

        Why not AirSim or Gazebo+PX4 SITL first?
        - AirSim is powerful but heavy, platform-sensitive, and slower to iterate for a browser-based demo.
        - Gazebo+PX4 SITL is excellent for real autopilot integration, but it requires ROS/Gazebo/PX4 toolchains and is not ideal for an instantly runnable Streamlit deliverable.
        - A custom simulator lets this deliverable run immediately, generate realistic synthetic telemetry/images, exercise planning/perception/safety logic, and produce a video without external services or API keys.

        The same mission, planning, perception, and safety interfaces are designed to transfer to PX4/ArduPilot SITL later.
        """
    )

    st.subheader("Safety documentation: human oversight and geofence")
    st.markdown(
        """
        ### Required human oversight
        - A certified remote operator or site safety officer must monitor the live telemetry dashboard during every flight.
        - The operator retains manual override authority via the ground station.
        - No flight may occur if aisles are occupied by personnel, forklifts, or temporary obstructions not present in the map.
        - Pre-flight checklist: battery >= 100%, propellers secure, camera/gimbal calibrated, AprilTags visible, dock contact clean, geofence loaded, failsafe tests passed.

        ### Geofence configuration
        - Horizontal geofence: warehouse outer boundary minus 0.5 m wall margin.
        - Vertical geofence: 0.1 m floor minimum, ceiling height minus 0.3 m maximum.
        - Shelf clearance envelope: all shelf obstacles inflated by the configured safe clearance (default 0.35 m).
        - No-fly zones: dock approach corridor only during takeoff/landing; charging contacts; fire exits; pedestrian walkways.
        - Failsafe geofence breach: immediate hover, then land in nearest designated safe area.

        ```json
        {
          "geofence": {
            "horizontal_margin_m": 0.5,
            "floor_min_m": 0.1,
            "ceiling_margin_m": 0.3,
            "shelf_inflation_m": 0.35,
            "safe_landing_areas": ["dock", "aisle_end_pad_A", "aisle_end_pad_B"],
            "rtl_battery_reserve_pct": 25
          }
        }
        ```
        """
    )

    st.subheader("README: transfer to a real drone")
    st.markdown(
        """
        ### PX4 / ArduPilot integration
        1. Replace the custom kinematic simulator with MAVSDK or pymavlink bridged to PX4/ArduPilot SITL.
        2. Map mission waypoints to MAVLink `MISSION_ITEM_INT` frames using local NED or ENU consistently.
        3. Use offboard mode for trajectory tracking; keep onboard failsafes enabled in parallel with ShelfScout safety monitors.
        4. Configure RTL/land failsafes in the autopilot to match the ShelfScout reserve threshold and safe landing areas.

        ### Camera calibration
        1. Calibrate intrinsic parameters using a ChArUco or AprilTag calibration board.
        2. Estimate extrinsic transform from camera optical frame to IMU/body frame.
        3. Validate gimbal pitch offset and timestamp synchronization between camera frames and VIO/AprilTag pose estimates.
        4. Record calibration version in every inventory report for traceability.

        ### Indoor flight-permit considerations
        1. Obtain site owner approval and align with warehouse safety manager.
        2. Confirm local regulations for indoor autonomous flight; many jurisdictions treat indoor flights differently from outdoor BVLOS rules.
        3. Define operational area, personnel exclusion windows, emergency stop procedure, and insurance requirements.
        4. Conduct a manual single-aisle checkout before enabling autonomous coverage planning.
        5. Keep a human observer with direct line of sight or redundant camera monitoring during initial deployments.
        """
    )

    st.subheader("Development order used in this demo")
    st.markdown(
        """
        1. Simulator + manual waypoint flight through one aisle.
        2. Coverage planner with TSP-like ordering and A* segments.
        3. Perception: detector, OCR, SKU validation, mismatch flags.
        4. Failsafes with deliberate simulated tests.
        5. Reports, replay video, and 30-mission evaluation.
        """
    )