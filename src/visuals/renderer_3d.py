from dataclasses import dataclass, field
import math
import random

import cv2
import numpy as np

try:
    from OpenGL.GL import GL_POINTS, glBegin, glColor3f, glEnd, glPointSize, glVertex3f

    OPENGL_AVAILABLE = True
except ImportError:
    GL_POINTS = None
    OPENGL_AVAILABLE = False


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
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0

current_rotation = np.zeros(3, dtype=np.float32)
target_rotation = np.zeros(3, dtype=np.float32)
target_rotation_step = np.array([1.0, 1.0, 0.0], dtype=np.float32)
rotation_lerp_factor = 0.1
scale_factor = 1.0
animation_time = 0.0
time_step = 0.08

position_x = 0
position_y = 0
viewport_width = 1
viewport_height = 1
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
    global target_rotation
    global scale_factor

    if gesture == "point":
        target_rotation[:] += target_rotation_step
    elif gesture == "peace":
        scale_factor = min(3.0, scale_factor + 0.02)
    elif gesture == "fist":
        target_rotation[:] = 0.0
        scale_factor = 1.0
    else:
        scale_factor = max(0.5, scale_factor - 0.01)


def update_animation():
    global animation_time

    animation_time += time_step
    _animate_particle_colors()


def update_position(x, y, frame_width, frame_height):
    global position_x
    global position_y
    global viewport_width
    global viewport_height

    if frame_width <= 0 or frame_height <= 0:
        return

    viewport_width = frame_width
    viewport_height = frame_height
    position_x = int(np.clip(x, 0, frame_width - 1))
    position_y = int(np.clip(y, 0, frame_height - 1))


def draw(frame):
    global viewport_width
    global viewport_height

    viewport_width = frame.shape[1]
    viewport_height = frame.shape[0]
    _update_current_rotation()
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


def render_particles_opengl():
    if not OPENGL_AVAILABLE:
        raise RuntimeError("PyOpenGL is not available in this environment.")

    _update_current_rotation()
    transformed_particles = _transform_particles(0.35 * scale_factor)
    offset_x, offset_y = _screen_to_opengl_offset()

    glPointSize(3)
    glBegin(GL_POINTS)

    for particle, point in zip(particles, transformed_particles):
        glColor3f(particle.r, particle.g, particle.b)
        glVertex3f(
            float(point[0] + offset_x),
            float(point[1] + offset_y),
            float(point[2]),
        )

    glEnd()


def _transform_particles(size):
    if not particles:
        return np.empty((0, 3), dtype=np.float32)

    points = np.array(
        [[particle.x, particle.y, particle.z] for particle in particles],
        dtype=np.float32,
    ) * size

    rotated = _rotate_x(points, math.radians(float(current_rotation[0])))
    rotated = _rotate_y(rotated, math.radians(float(current_rotation[1])))
    rotated = _rotate_z(rotated, math.radians(float(current_rotation[2])))
    return rotated


def _update_current_rotation():
    current_rotation[:] += (target_rotation - current_rotation) * rotation_lerp_factor


def _animate_particle_colors():
    for particle in particles:
        offset = (particle.x * 1.7) + (particle.y * 2.3) + (particle.z * 2.9)
        particle.r = _normalized_sine(animation_time + offset)
        particle.g = _normalized_sine(animation_time + offset + 2.0)
        particle.b = _normalized_sine(animation_time + offset + 4.0)


def _normalized_sine(value):
    return (math.sin(value) + 1.0) * 0.5


def _project_particles(center):
    if not particles:
        return []

    rotated = _transform_particles(45.0 * scale_factor)

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
                (
                    int(particle.b * 255),
                    int(particle.g * 255),
                    int(particle.r * 255),
                ),
            )
        )

    projected.sort(key=lambda point: point[2], reverse=True)
    return projected


def _screen_to_opengl_offset():
    if viewport_width <= 0 or viewport_height <= 0:
        return 0.0, 0.0

    center_x = position_x if position_x else viewport_width / 2
    center_y = position_y if position_y else viewport_height / 2
    offset_x = ((center_x / viewport_width) * 2.0 - 1.0) * 0.75
    offset_y = (1.0 - (center_y / viewport_height) * 2.0) * 0.75
    return float(offset_x), float(offset_y)


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
        (
            "Rot: "
            f"({current_rotation[0]:.1f}, {current_rotation[1]:.1f}, "
            f"{current_rotation[2]:.1f})"
        ),
        (20, frame.shape[0] - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        (
            "Target: "
            f"({target_rotation[0]:.1f}, {target_rotation[1]:.1f}, "
            f"{target_rotation[2]:.1f})"
        ),
        (20, frame.shape[0] - 105),
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
    cv2.putText(
        frame,
        f"OpenGL: {'ready' if OPENGL_AVAILABLE else 'unavailable'}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Time: {animation_time:.2f}",
        (20, 100),
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
