from dataclasses import dataclass, field
import math
import random

import cv2
import numpy as np


def _noop_gl(*_args, **_kwargs):
    return None


try:
    from OpenGL.GL import GL_POINTS, glBegin, glColor3f, glEnd, glPointSize, glVertex3f

    OPENGL_AVAILABLE = True
except ImportError:
    GL_POINTS = 0
    glBegin = _noop_gl
    glColor3f = _noop_gl
    glEnd = _noop_gl
    glPointSize = _noop_gl
    glVertex3f = _noop_gl
    OPENGL_AVAILABLE = False


def _random_velocity():
    return random.uniform(-0.012, 0.012)


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
    home_x: float = field(init=False)
    home_y: float = field(init=False)
    home_z: float = field(init=False)

    def __post_init__(self):
        self.home_x = self.x
        self.home_y = self.y
        self.home_z = self.z

current_rotation = np.zeros(3, dtype=np.float32)
target_rotation = np.zeros(3, dtype=np.float32)
rotation_step = np.array([0.45, 0.7, 0.18], dtype=np.float32)
rotation_lerp_factor = 0.1
rotation_speed_multiplier = 0.35
target_rotation_speed_multiplier = 0.35
rotation_speed_lerp_factor = 0.14
scale_factor = 1.0
target_scale_factor = 1.0
scale_lerp_factor = 0.12
animation_time = 0.0
time_step = 0.08
color_mode = "rainbow"
motion_mode = "orbit"
active_gesture = "idle"
velocity_jitter = 0.0025
attraction_strength = 0.018
velocity_damping = 0.94
max_velocity = 0.05
expand_distance_multiplier = 1.75
free_float_boundary = 3.2
free_float_pull_strength = 0.018

position_x = 0
position_y = 0
viewport_width = 1
viewport_height = 1
particle_spacing = 0.2
particles: list[Particle] = []
trail_fade_factor = 0.86
trail_blend_strength = 0.82
trail_glow_radius = 3
trail_buffer: np.ndarray | None = None
trail_clear_buffer: np.ndarray | None = None


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
    global active_gesture
    global color_mode
    global motion_mode
    global target_rotation
    global target_rotation_speed_multiplier
    global target_scale_factor

    gesture_changed = gesture != active_gesture
    active_gesture = gesture

    if gesture == "point":
        target_rotation_speed_multiplier = 2.6
        target_scale_factor = 1.0
        motion_mode = "orbit"
        color_mode = "blue"
        if gesture_changed:
            target_rotation[:] += rotation_step * 14.0
    elif gesture == "peace":
        target_rotation_speed_multiplier = 0.85
        target_scale_factor = 1.15
        motion_mode = "expand"
        color_mode = "rainbow"
        if gesture_changed:
            _apply_radial_impulse(0.035)
    elif gesture == "fist":
        target_rotation_speed_multiplier = 0.12
        target_scale_factor = 0.82
        motion_mode = "collapse"
        color_mode = "white"
        if gesture_changed:
            _apply_radial_impulse(-0.05)
    elif gesture == "open":
        target_rotation_speed_multiplier = 1.2
        target_scale_factor = 1.05
        motion_mode = "free"
        color_mode = "shift"
        if gesture_changed:
            _apply_random_impulse(0.03)
    else:
        target_rotation_speed_multiplier = 0.35
        target_scale_factor = 1.0
        motion_mode = "orbit"
        color_mode = "rainbow"


def update_animation():
    global animation_time

    animation_time += time_step
    _update_rotation_state()
    _update_scale_factor()
    _update_particle_motion()
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
    _ensure_trail_buffers(frame)
    _fade_trail_buffer()
    _draw_trail_particles(projected_particles)
    _draw_shadow(frame, center)
    frame = _blend_trail_buffer(frame)
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


def _update_rotation_state():
    global rotation_speed_multiplier

    rotation_speed_multiplier += (
        target_rotation_speed_multiplier - rotation_speed_multiplier
    ) * rotation_speed_lerp_factor
    target_rotation[:] += rotation_step * rotation_speed_multiplier


def _update_scale_factor():
    global scale_factor

    scale_factor += (target_scale_factor - scale_factor) * scale_lerp_factor


def _update_particle_motion():
    for particle in particles:
        jitter = velocity_jitter
        attraction = attraction_strength
        damping = velocity_damping
        speed_limit = max_velocity
        target_x = particle.home_x
        target_y = particle.home_y
        target_z = particle.home_z

        if motion_mode == "expand":
            target_x = particle.home_x * expand_distance_multiplier
            target_y = particle.home_y * expand_distance_multiplier
            target_z = particle.home_z * expand_distance_multiplier
            attraction *= 1.2
            jitter *= 1.35
            damping = 0.95
            speed_limit *= 1.5
        elif motion_mode == "collapse":
            target_x = 0.0
            target_y = 0.0
            target_z = 0.0
            attraction *= 2.4
            jitter *= 0.6
            damping = 0.9
            speed_limit *= 1.25
        elif motion_mode == "free":
            attraction = 0.0
            jitter *= 1.9
            damping = 0.97
            speed_limit *= 1.75

        particle.vx += random.uniform(-jitter, jitter)
        particle.vy += random.uniform(-jitter, jitter)
        particle.vz += random.uniform(-jitter, jitter)

        if attraction > 0.0:
            particle.vx += (target_x - particle.x) * attraction
            particle.vy += (target_y - particle.y) * attraction
            particle.vz += (target_z - particle.z) * attraction

        if motion_mode == "free":
            _apply_free_float_boundary(particle)

        particle.vx *= damping
        particle.vy *= damping
        particle.vz *= damping

        speed = math.sqrt(
            (particle.vx * particle.vx)
            + (particle.vy * particle.vy)
            + (particle.vz * particle.vz)
        )
        if speed > speed_limit:
            speed_scale = speed_limit / speed
            particle.vx *= speed_scale
            particle.vy *= speed_scale
            particle.vz *= speed_scale

        particle.x += particle.vx
        particle.y += particle.vy
        particle.z += particle.vz


def _apply_radial_impulse(strength):
    for particle in particles:
        distance = math.sqrt(
            (particle.x * particle.x)
            + (particle.y * particle.y)
            + (particle.z * particle.z)
        )
        if distance == 0.0:
            continue

        particle.vx += (particle.x / distance) * strength
        particle.vy += (particle.y / distance) * strength
        particle.vz += (particle.z / distance) * strength


def _apply_random_impulse(strength):
    for particle in particles:
        particle.vx += random.uniform(-strength, strength)
        particle.vy += random.uniform(-strength, strength)
        particle.vz += random.uniform(-strength, strength)


def _apply_free_float_boundary(particle):
    distance = math.sqrt(
        (particle.x * particle.x)
        + (particle.y * particle.y)
        + (particle.z * particle.z)
    )
    if distance <= free_float_boundary or distance == 0.0:
        return

    pull_strength = (distance - free_float_boundary) * free_float_pull_strength
    particle.vx -= (particle.x / distance) * pull_strength
    particle.vy -= (particle.y / distance) * pull_strength
    particle.vz -= (particle.z / distance) * pull_strength


def _animate_particle_colors():
    if color_mode == "blue":
        _set_solid_particle_color(0.2, 0.45, 1.0)
        return

    if color_mode == "white":
        _set_solid_particle_color(1.0, 1.0, 1.0)
        return

    if color_mode == "shift":
        _set_dynamic_shift_colors()
        return

    _set_rainbow_colors()


def _set_solid_particle_color(r, g, b):
    for particle in particles:
        particle.r = r
        particle.g = g
        particle.b = b


def _set_rainbow_colors():
    for particle in particles:
        offset = (particle.x * 1.7) + (particle.y * 2.3) + (particle.z * 2.9)
        particle.r = _normalized_sine(animation_time + offset)
        particle.g = _normalized_sine(animation_time + offset + 2.0)
        particle.b = _normalized_sine(animation_time + offset + 4.0)


def _set_dynamic_shift_colors():
    for particle in particles:
        wave_x = animation_time * 1.4 + particle.x * 3.2
        wave_y = animation_time * 1.1 + particle.y * 3.8
        wave_z = animation_time * 1.7 + particle.z * 4.4
        particle.r = _normalized_sine(wave_x + wave_z)
        particle.g = _normalized_sine(wave_y + 2.0)
        particle.b = _normalized_sine(wave_z - wave_x + 4.0)


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


def _ensure_trail_buffers(frame):
    global trail_buffer
    global trail_clear_buffer

    if trail_buffer is not None and trail_buffer.shape == frame.shape:
        return

    # Keep a dedicated particle layer so old positions can fade out gradually.
    trail_buffer = np.zeros_like(frame)
    trail_clear_buffer = np.zeros_like(frame)


def _fade_trail_buffer():
    global trail_buffer

    if trail_buffer is None or trail_clear_buffer is None:
        return

    trail_buffer[:] = cv2.addWeighted(
        trail_buffer,
        trail_fade_factor,
        trail_clear_buffer,
        1.0 - trail_fade_factor,
        0.0,
    )


def _blend_trail_buffer(frame):
    if trail_buffer is None:
        return frame

    return cv2.addWeighted(frame, 1.0, trail_buffer, trail_blend_strength, 0.0)


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


def _draw_trail_particles(projected_particles):
    if trail_buffer is None:
        return

    frame_height, frame_width = trail_buffer.shape[:2]

    for x, y, _z, radius, color in projected_particles:
        if 0 <= x < frame_width and 0 <= y < frame_height:
            glow_color = tuple(min(255, int(channel * 0.45)) for channel in color)
            cv2.circle(
                trail_buffer,
                (x, y),
                radius + trail_glow_radius,
                glow_color,
                -1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                trail_buffer,
                (x, y),
                radius + 1,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )


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
    cv2.putText(
        frame,
        f"Color: {color_mode}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        (
            "Effect: "
            f"{motion_mode} | Spin: {rotation_speed_multiplier:.2f}x"
        ),
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        (
            "Motion: "
            f"jitter {velocity_jitter:.4f} | attract {attraction_strength:.3f}"
        ),
        (20, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        (
            "Trails: "
            f"fade {trail_fade_factor:.2f} | mix {trail_blend_strength:.2f}"
        ),
        (20, 220),
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
