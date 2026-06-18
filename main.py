"""
F1 Top-Down Racing Simulation
Entry point with keyboard control, wall collisions, lap timing, and sensors.
Press T to switch between tracks!
"""
import pygame
import numpy as np
import time
import sys
import os

sys.path.insert(0, 'src')

from config import SIM, RACE, RENDER, TRACK, SENSOR
from src.physics.world import World
from src.physics.car import Car
from src.track.track import Track
from src.rendering.renderer import Renderer
from src.sensors.sensor import RayCaster, FrenetObserver


TRACK_CATALOG = [
    ("Sprint Circuit",     Track.create_sprint_track),
    ("Grand Prix Circuit", Track.create_complex_track),
]


def _spawn_player_cars(world, track, n):
    cars = []
    spawn_gap = max(RACE.player_spawn_gap, RACE.min_safe_spawn_gap)
    for i in range(n):
        s_i = (-i * spawn_gap) % track.total_length
        pos, heading, _ = track.get_pose_at_s(s_i)
        cars.append(Car(world, position=pos, angle=heading, car_id=i, is_main_player=(i == 0)))
    return cars


def _spawn_static_control_car(world, track, start_id):
    # place control car AHEAD of start line
    s_ctrl = RACE.static_control_spawn_ahead % track.total_length
    pos, heading, _ = track.get_pose_at_s(s_ctrl)
    car = Car(world, position=pos, angle=heading, car_id=start_id, is_static_control=True)
    car.body.type = 2  # dynamic (kept in physics); controlled kinematically below
    return car


def load_track(track_idx, renderer):
    """
    Build a fresh world + track + car for the chosen track.
    Adjusts camera zoom to fit. Returns everything the main loop needs.
    """
    name, creator = TRACK_CATALOG[track_idx]
    print(f"\nLoading: {name}...")

    world = World()
    track = creator(track_width=22)
    print(f"  Length: {track.total_length:.0f}m ({track.total_length/1000:.1f}km)")
    print(f"  Centerline: {len(track.centerline)} pts")

    track.create_walls(world)

    cars = _spawn_player_cars(world, track, RACE.num_players)
    static_control_car = _spawn_static_control_car(world, track, len(cars)) if RACE.enable_static_control_car else None
    all_cars = cars + ([static_control_car] if static_control_car is not None else [])
    inner, outer = track.get_boundary_points()
    checkpoints = track.get_checkpoint_positions()
    print(f"  {len(checkpoints)} sector lines")

    start_pos = track.centerline[0]
    start_heading = np.arctan2(track.tangents[0, 1], track.tangents[0, 0])

    track_span = np.max(track.centerline, axis=0) - np.min(track.centerline, axis=0)
    max_span = max(track_span)
    renderer.zoom = (min(RENDER.screen_width, RENDER.screen_height)
                     / (max_span + 50) / SIM.pixels_per_meter)

    world.collision_handler.reset()
    world.collision_handler.ignore_car_collision_count_until_step = RACE.startup_collision_grace_steps
    return (world, track, cars, static_control_car, all_cars, inner, outer, checkpoints,
            start_pos, start_heading, name)


def handle_input(keys, car):
    throttle = 0.0
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        throttle = 1.0
    elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
        throttle = -0.5

    steering = 0.0
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        steering = 1.0
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        steering = -1.0

    car.set_controls(throttle, steering)


def _reset_race_state(world, track, cars, static_control_car):
    spawn_gap = max(RACE.player_spawn_gap, RACE.min_safe_spawn_gap)
    for idx, c in enumerate(cars):
        s_i = (-idx * spawn_gap) % track.total_length
        pos_i, heading_i, _ = track.get_pose_at_s(s_i)
        c.body.position = (float(pos_i[0]), float(pos_i[1]))
        c.body.angle = float(heading_i)
        c.body.linearVelocity = (0.0, 0.0)
        c.body.angularVelocity = 0.0

    static_s = RACE.static_control_spawn_ahead % track.total_length
    if static_control_car is not None:
        p, h, _ = track.get_pose_at_s(static_s)
        static_control_car.body.position = (float(p[0]), float(p[1]))
        static_control_car.body.angle = float(h)
        t = np.array([np.cos(h), np.sin(h)]) * RACE.static_control_speed
        static_control_car.body.linearVelocity = (float(t[0]), float(t[1]))
        static_control_car.body.angularVelocity = 0.0

    world.collision_handler.reset()
    world.collision_handler.ignore_car_collision_count_until_step = RACE.startup_collision_grace_steps
    return static_s


def main():
    print("=" * 50)
    print("F1 Top-Down Racing Simulation - Phase 5")
    print("=" * 50)

    # --- Renderer (persists across track switches) ---
    renderer = Renderer()

    # --- Sensors (persist across track switches) ---
    raycaster = RayCaster(
        num_forward_rays=SENSOR.num_forward_rays,
        forward_spread=SENSOR.forward_spread,
        num_mirror_rays=SENSOR.num_mirror_rays,
        mirror_start=SENSOR.mirror_angle_start,
        mirror_end=SENSOR.mirror_angle_end,
        max_distance=SENSOR.max_ray_distance
    )
    frenet_observer = FrenetObserver(
        num_lookahead=SENSOR.num_lookahead,
        lookahead_spacing=SENSOR.lookahead_spacing
    )
    print(f"\nSensors: {raycaster.num_forward} fwd + "
          f"{raycaster.num_mirror_per_side * 2} mirror = {raycaster.num_rays} rays, "
          f"{SENSOR.max_ray_distance:.0f}m range")

    # --- Load initial track ---
    track_idx = 1  # Start with GP circuit
    (world, track, cars, static_control_car, all_cars, inner_boundary, outer_boundary, checkpoints,
     start_pos, start_heading, track_name) = load_track(track_idx, renderer)
    car = cars[0]
    static_s = _reset_race_state(world, track, cars, static_control_car)

    print("\n" + "=" * 50)
    print("Controls:")
    print("  Up/W: Accelerate       Down/S: Brake/Reverse")
    print("  Left/A: Steer Left    Right/D: Steer Right")
    print("  R: Reset car   C: Camera   V: Sensors")
    print("  T: Track       P: Screenshot")
    print("  +/-: Zoom      ESC: Quit")
    print("=" * 50 + "\n")

    # --- Timing state ---
    prev_s = 0.0
    lap_start_time = time.time()
    current_lap_time = 0.0
    best_lap_time = float('inf')
    lap_count = 0
    current_sector = 0

    # --- Flags ---
    running = True
    follow_camera = True
    show_sensors = True

    # ======================== MAIN LOOP ========================
    while running:
        # --- Events ---
        for event in renderer.get_events():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    static_s = _reset_race_state(world, track, cars, static_control_car)
                    prev_s = 0.0
                    lap_start_time = time.time()
                    current_lap_time = 0.0
                    lap_count = 0
                    current_sector = 0
                    print("Car reset to start position")

                elif event.key == pygame.K_t:
                    track_idx = (track_idx + 1) % len(TRACK_CATALOG)
                    (world, track, cars,
                     static_control_car, all_cars, inner_boundary, outer_boundary, checkpoints,
                     start_pos, start_heading, track_name
                     ) = load_track(track_idx, renderer)
                    car = cars[0]
                    static_s = _reset_race_state(world, track, cars, static_control_car)
                    prev_s = 0.0
                    lap_start_time = time.time()
                    current_lap_time = 0.0
                    best_lap_time = float('inf')
                    lap_count = 0
                    current_sector = 0

                elif event.key == pygame.K_c:
                    follow_camera = not follow_camera
                    print(f"Camera follow: {'ON' if follow_camera else 'OFF'}")
                elif event.key == pygame.K_v:
                    show_sensors = not show_sensors
                    print(f"Sensors display: {'ON' if show_sensors else 'OFF'}")
                elif event.key == pygame.K_p:
                    os.makedirs('screenshots', exist_ok=True)
                    fname = f"screenshots/{track_name.replace(' ','_')}_{int(time.time())}.png"
                    pygame.image.save(renderer.screen, fname)
                    print(f"Screenshot saved: {fname}")
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    renderer.zoom *= 1.2
                elif event.key == pygame.K_MINUS:
                    renderer.zoom /= 1.2

        # --- Input ---
        keys = pygame.key.get_pressed()
        handle_input(keys, cars[0])

        for i in range(1, len(cars)):
            cars[i].set_controls(0.0, 0.0)

        for c in cars:
            c.update()

        if static_control_car is not None:
            static_s = (static_s + RACE.static_control_speed * SIM.time_step) % track.total_length
            p, h, _ = track.get_pose_at_s(static_s)
            static_control_car.body.position = (float(p[0]), float(p[1]))
            static_control_car.body.angle = float(h)
            t = np.array([np.cos(h), np.sin(h)]) * RACE.static_control_speed
            static_control_car.body.linearVelocity = (float(t[0]), float(t[1]))
            static_control_car.body.angularVelocity = 0.0

        world.step()

        frenet = track.get_frenet_coordinates(cars[0].position, cars[0].angle)
        s = frenet['s']

        # --- Sensors ---
        ray_kwargs = {}
        if SENSOR.detect_cars_as_obstacles:
            ray_kwargs = {"cars": all_cars, "ego_car": cars[0]}
        ray_distances, ray_hits = raycaster.cast(
            cars[0].position, cars[0].angle, inner_boundary, outer_boundary, **ray_kwargs
        )

        # --- Lap timing ---
        if prev_s > track.total_length * 0.8 and s < track.total_length * 0.2:
            if lap_count > 0 or current_lap_time > 5.0:
                finished_time = time.time() - lap_start_time
                if finished_time < best_lap_time:
                    best_lap_time = finished_time
                    print(f"  * NEW BEST LAP: {best_lap_time:.2f}s")
                else:
                    print(f"  Lap {lap_count + 1}: {finished_time:.2f}s "
                          f"(best: {best_lap_time:.2f}s)")
                lap_count += 1
            lap_start_time = time.time()

        current_lap_time = time.time() - lap_start_time
        prev_s = s

        sector_length = track.total_length / TRACK.num_sectors
        current_sector = int(s / sector_length) % TRACK.num_sectors

        # wall-contact display should reflect ONLY the main player car
        player_stats = world.collision_handler.get_car_stats(cars[0].car_id)
        touching_wall = player_stats['touching_wall']
        player_wall_hits = player_stats['wall_hit_count']
        player_car_hits = player_stats['car_collision_count']

        # --- Camera ---
        if follow_camera:
            renderer.set_camera(cars[0].position[0], cars[0].position[1])
        else:
            track_center = np.mean(track.centerline, axis=0)
            renderer.set_camera(track_center[0], track_center[1])

        # --- Render ---
        renderer.clear()
        renderer.draw_track(track)
        renderer.draw_checkpoints(checkpoints)
        renderer.draw_cars(all_cars, collision_handler=world.collision_handler)
        renderer.draw_frenet_debug(cars[0], frenet)

        if show_sensors:
            renderer.draw_rays(cars[0].position, ray_distances, ray_hits,
                               raycaster.max_distance, raycaster.is_mirror)
            renderer.draw_sensor_panel(ray_distances, raycaster.max_distance,
                                       raycaster.is_mirror)

        renderer.draw_track_name(track_name, track.total_length)
        renderer.draw_hud(cars[0], frenet)
        renderer.draw_lap_timer({
            'current_time': current_lap_time,
            'best_time': best_lap_time,
            'lap_count': lap_count,
            'wall_hits': player_wall_hits,   # player-only
            'touching_wall': touching_wall,  # player-only
            'current_sector': current_sector,
            'car_hits': player_car_hits,     # player-only
        })

        renderer.update()
        renderer.tick(60)

    # --- Cleanup ---
    renderer.quit()
    print("\nSimulation ended.")
    if best_lap_time < float('inf'):
        print(f"Best lap time: {best_lap_time:.2f}s")
    final_player_stats = world.collision_handler.get_car_stats(cars[0].car_id)
    print(f"Total wall hits (player): {final_player_stats['wall_hit_count']}")


if __name__ == "__main__":
    main()
