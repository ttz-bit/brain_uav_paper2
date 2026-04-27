from __future__ import annotations

import math
from collections.abc import Mapping

from paper2.render.coordinate_mapper import WorldState


def sample_mode(mode_probs: Mapping[str, float], rng) -> str:
    modes = list(mode_probs.keys())
    probs = [float(mode_probs[k]) for k in modes]
    s = sum(probs)
    if s <= 0:
        probs = [1.0 / len(modes)] * len(modes)
    else:
        probs = [p / s for p in probs]
    idx = int(rng.choice(len(modes), p=probs))
    return modes[idx]


def generate_motion_sequence(
    mode: str,
    frames: int,
    dt: float,
    world_size_m: float,
    rng,
) -> list[WorldState]:
    x = float(rng.uniform(0.2, 0.8) * world_size_m)
    y = float(rng.uniform(0.2, 0.8) * world_size_m)
    speed = float(rng.uniform(8.0, 24.0))
    heading = float(rng.uniform(-math.pi, math.pi))
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    seq: list[WorldState] = []

    turn_rate = float(rng.uniform(-0.06, 0.06))
    switch_every = int(rng.integers(4, 9))
    evasive_gain = float(rng.uniform(0.12, 0.22))

    for t in range(frames):
        if mode == "turn":
            heading += turn_rate * dt
            vx = speed * math.cos(heading)
            vy = speed * math.sin(heading)
        elif mode == "piecewise":
            if t > 0 and t % switch_every == 0:
                heading += float(rng.uniform(-0.8, 0.8))
                vx = speed * math.cos(heading)
                vy = speed * math.sin(heading)
        elif mode == "evasive":
            heading += evasive_gain * math.sin(0.6 * t) * dt
            speed = max(6.0, min(26.0, speed + float(rng.uniform(-1.0, 1.0))))
            vx = speed * math.cos(heading)
            vy = speed * math.sin(heading)

        x += vx * dt
        y += vy * dt

        if x < 0 or x > world_size_m:
            vx *= -1.0
            x = min(world_size_m, max(0.0, x))
            heading = math.atan2(vy, vx)
        if y < 0 or y > world_size_m:
            vy *= -1.0
            y = min(world_size_m, max(0.0, y))
            heading = math.atan2(vy, vx)

        seq.append(WorldState(x=x, y=y, vx=vx, vy=vy, heading=heading))

    return seq
