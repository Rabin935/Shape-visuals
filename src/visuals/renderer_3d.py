from dataclasses import dataclass, field
import math
import random

import cv2
import numpy as np


def _random_velocity():
    return random.uniform(-0.35, 0.35)


@dataclass(slots=True)
class Particle:
    x: float
    y: float
    z: float
    vx: float = field(default_factory=_random_velocity)
    vy: float = field(default_factory=_random_velocity)
    vz: float = field(default_factory=_random_velocity)
    r: int = 255
    g: int = 255
    b: int = 255


rotation_x = 0.0
rotation_y = 0.0
rotation_z = 0.0
scale_factor = 1.0

position_x = 0
position_y = 0
particle_spacing = 0.2
particles: list[Particle] = []


def generate_cube(spacing=0.2):
    global particle_spacing
    global particles

    if spacing <= 0:
        raise ValueError("spacing must be greater than 0")

    particle_spacing = spacing
    particles = []
    stop = 1.0 + (spacing * 0.5)

    for x in np.arange(-1.0, stop, spacing):
        for y in np.arange(-1.0, stop, spacing):
            for z in np.arange(-1.0, stop, spacing):
                particles.append(Particle(float(x), float(y), float(z)))

    return particles


generate_cube()


def update_transform(gesture):
    global rotation_x
    global rotation_y
    global rotation_z
    global scale_factor

    if gesture == "point":
        rotation_x += 1.0
        rotation_y += 1.0
    elif gesture == "peace":
        scale_factor = min(3.0, scale_factor + 0.02)
    elif gesture == "fist":
        rotation_x = 0.0
        rotation_y = 0.0
        rotation_z = 0.0
        scale_factor = 1.0
    else:
        scale_factor = max(0.5, scale_factor - 0.01)


def update_position(x, y, frame_width, frame_height):
    global position_x
    global position_y

    if frame_width <= 0 or frame_height <= 0:
        return

    position_x = int(np.clip(x, 0, frame_width - 1))
    position_y = int(np.clip(y, 0, frame_height - 1))


def draw(frame):
    center = np.array(
        [
            position_x if position_x else frame.shape[1] // 2,
            position_y if position_y else frame.shape[0] // 2,
        ],
        dtype=np.float32,
    )

    projected_particles = _project_particles(center)
    _draw_shadow(frame, center)
    _draw_particles(frame, projected_particles)
    _draw_debug(frame)
    return frame


def _project_particles(center):
    if not particles:
        return []

    half_size = 45.0 * scale_factor
    points = np.array(
        [[particle.x, particle.y, particle.z] for particle in particles],
        dtype=np.float32,
    ) * half_size

    rotated = _rotate_x(points, math.radians(rotation_x))
    rotated = _rotate_y(rotated, math.radians(rotation_y))
    rotated = _rotate_z(rotated, math.radians(rotation_z))

    perspective = 320.0
    depth_offset = 260.0
    projected = []

    for particle, point in zip(particles, rotated):
        z = point[2] + depth_offset
        factor = perspective / z
        x = center[0] + point[0] * factor
        y = center[1] + point[1] * factor
        radius = max(1, int(2.2 * factor))
        projected.append(
            (
                int(x),
                int(y),
                z,
                radius,
                (particle.b, particle.g, particle.r),
            )
        )

    projected.sort(key=lambda point: point[2], reverse=True)
    return projected


def _draw_shadow(frame, center):
    shadow_center = (int(center[0] + 28), int(center[1] + 62))
    axes = (
        int(max(24, 55 * scale_factor)),
        int(max(10, 18 * scale_factor)),
    )
    cv2.ellipse(frame, shadow_center, axes, 0, 0, 360, (25, 25, 25), -1)


def _draw_particles(frame, projected_particles):
    frame_height, frame_width = frame.shape[:2]

    for x, y, _z, radius, color in projected_particles:
        if 0 <= x < frame_width and 0 <= y < frame_height:
            cv2.circle(frame, (x, y), radius, color, -1, lineType=cv2.LINE_AA)


def _draw_debug(frame):
    cv2.putText(
        frame,
        f"Particles: {len(particles)} | Spacing: {particle_spacing:.2f}",
        (20, frame.shape[0] - 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Rot: ({rotation_x:.0f}, {rotation_y:.0f}, {rotation_z:.0f})",
        (20, frame.shape[0] - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Scale: {scale_factor:.2f}",
        (20, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _rotate_x(points, angle):
    matrix = np.array(
        [
            [1, 0, 0],
            [0, math.cos(angle), -math.sin(angle)],
            [0, math.sin(angle), math.cos(angle)],
        ],
        dtype=np.float32,
    )
    return points @ matrix.T


def _rotate_y(points, angle):
    matrix = np.array(
        [
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)],
        ],
        dtype=np.float32,
    )
    return points @ matrix.T


def _rotate_z(points, angle):
    matrix = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    return points @ matrix.T
