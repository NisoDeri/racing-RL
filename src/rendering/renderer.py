"""
Pygame Renderer
Visualizes the track, car, checkpoints, collision feedback, and lap timing.
"""
import pygame
import numpy as np
import sys
sys.path.append('..')
from config import RENDER, SIM


# Sector colors (like real F1 timing: S1=red, S2=blue, S3=yellow)
SECTOR_COLORS = [
    (220, 50, 50),    # Sector 1 — red
    (50, 120, 220),   # Sector 2 — blue
    (220, 200, 50),   # Sector 3 — yellow
]


class Renderer:
    """
    Renders the simulation using Pygame.
    Handles coordinate transformation between physics (meters) and screen (pixels).
    """
    
    def __init__(self):
        """Initialize Pygame and create the window."""
        pygame.init()
        self.screen = pygame.display.set_mode(
            (RENDER.screen_width, RENDER.screen_height)
        )
        pygame.display.set_caption("F1 Top-Down Racing")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_big = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 20)
        
        # Camera position (center of view in world coordinates)
        self.camera_x = 0
        self.camera_y = 0
        self.zoom = RENDER.zoom
        
        # Collision flash timer (frames remaining)
        self._flash_frames = 0

    def world_to_screen(self, world_pos):
        """
        Convert world coordinates (meters) to screen coordinates (pixels).
        
        Args:
            world_pos: (x, y) in meters
            
        Returns:
            (screen_x, screen_y) in pixels
        """
        # Apply camera offset
        x = world_pos[0] - self.camera_x
        y = world_pos[1] - self.camera_y
        
        # Scale to pixels and apply zoom
        scale = SIM.pixels_per_meter * self.zoom
        screen_x = RENDER.screen_width / 2 + x * scale
        screen_y = RENDER.screen_height / 2 - y * scale  # Y is flipped
        
        return int(screen_x), int(screen_y)
    
    def meters_to_pixels(self, meters):
        """Convert a distance from meters to pixels."""
        return int(meters * SIM.pixels_per_meter * self.zoom)
    
    def set_camera(self, x, y):
        """Set camera center position in world coordinates."""
        self.camera_x = x
        self.camera_y = y
    
    def clear(self):
        """Clear the screen with background color."""
        self.screen.fill(RENDER.background_color)
    
    def draw_track(self, track):
        """
        Draw the track (centerline and boundaries).
        
        Args:
            track: Track instance
        """
        # Get boundary points
        inner, outer = track.get_boundary_points()
        
        # Convert to screen coordinates
        inner_screen = [self.world_to_screen(p) for p in inner]
        outer_screen = [self.world_to_screen(p) for p in outer]
        center_screen = [self.world_to_screen(p) for p in track.centerline]
        
        # Draw track surface (filled polygon between boundaries)
        # Create a polygon by combining outer + reversed inner
        track_polygon = outer_screen + inner_screen[::-1]
        if len(track_polygon) > 2:
            pygame.draw.polygon(self.screen, RENDER.track_color, track_polygon)
        
        # Draw boundaries (white lines)
        if len(inner_screen) > 1:
            pygame.draw.lines(self.screen, RENDER.track_border_color, True, inner_screen, 2)
            pygame.draw.lines(self.screen, RENDER.track_border_color, True, outer_screen, 2)
        
        # Draw centerline (dashed yellow) - simplified as solid for now
        if len(center_screen) > 1:
            pygame.draw.lines(self.screen, RENDER.centerline_color, True, center_screen, 1)
    
    def draw_checkpoints(self, checkpoints):
        """
        Draw sector/checkpoint lines across the track.
        
        Start/finish line = wider white-red pattern.
        Sector lines = colored lines matching F1 sectors.
        
        Args:
            checkpoints: list of checkpoint dicts from track.get_checkpoint_positions()
        """
        for cp in checkpoints:
            inner_scr = self.world_to_screen(cp['inner'])
            outer_scr = self.world_to_screen(cp['outer'])
            
            if cp['is_start_finish']:
                # Start/finish line — wide, white+red
                pygame.draw.line(self.screen, (255, 255, 255), inner_scr, outer_scr, 4)
                # Red line slightly offset for checkered effect
                mid = ((inner_scr[0] + outer_scr[0]) // 2,
                       (inner_scr[1] + outer_scr[1]) // 2)
                pygame.draw.line(self.screen, (255, 50, 50), inner_scr, mid, 3)
            else:
                # Sector line — colored, thinner
                color_idx = cp['index'] % len(SECTOR_COLORS)
                pygame.draw.line(self.screen, SECTOR_COLORS[color_idx], 
                               inner_scr, outer_scr, 2)
    
    def draw_car(self, car, touching_wall=False, touching_car=False):
        """
        Draw the car as a rectangle.
        Outline flashes red when touching a wall.
        
        Args:
            car: Car instance
            touching_wall: whether car is currently in contact with a wall
        """
        from config import CAR
        
        # Get car corners in world coordinates
        pos = car.position
        angle = car.angle
        half_length = CAR.length / 2
        half_width = CAR.width / 2
        
        # Car corners relative to center (before rotation)
        corners = np.array([
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width]
        ])
        
        # Rotate corners
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        rotated_corners = [rotation_matrix @ c + pos for c in corners]
        
        # Convert to screen coordinates
        screen_corners = [self.world_to_screen(c) for c in rotated_corners]
        
        # Draw car body
        body_color = RENDER.car_color
        if getattr(car, "is_static_control", False):
            body_color = (70, 170, 255)
        elif getattr(car, "is_main_player", False):
            body_color = (230, 50, 50)
        else:
            body_color = (240, 150, 50)

        pygame.draw.polygon(self.screen, body_color, screen_corners)

        if touching_wall:
            outline_color, outline_width = (255, 50, 50), 3
        elif touching_car:
            outline_color, outline_width = (255, 0, 255), 3
        elif self._flash_frames > 0:
            outline_color = (255, 120, 50)  # Fading orange
            outline_width = 2
            self._flash_frames -= 1
        else:
            outline_color = (255, 255, 255)
            outline_width = 2
        
        pygame.draw.polygon(self.screen, outline_color, screen_corners, outline_width)
        
        # Draw front indicator (small triangle at front)
        front_center = (rotated_corners[0] + rotated_corners[1]) / 2
        front_screen = self.world_to_screen(front_center)
        pygame.draw.circle(self.screen, (255, 255, 0), front_screen, 5)

    def draw_point(self, position, color=(255, 0, 0), radius=5):
        """Draw a simple point (for debugging)."""
        screen_pos = self.world_to_screen(position)
        pygame.draw.circle(self.screen, color, screen_pos, radius)
    
    def draw_line(self, start, end, color=(255, 255, 255), width=1):
        """Draw a line between two world positions."""
        start_screen = self.world_to_screen(start)
        end_screen = self.world_to_screen(end)
        pygame.draw.line(self.screen, color, start_screen, end_screen, width)
    
    def draw_frenet_debug(self, car, frenet_data):
        """
        Draw debug visualization of Frenet frame data.
        Shows projected point and lateral deviation.
        
        Args:
            car: Car instance
            frenet_data: Dictionary from track.get_frenet_coordinates()
        """
        # Draw projected point on centerline
        projected = frenet_data['projected']
        self.draw_point(projected, color=(0, 255, 0), radius=4)
        
        # Draw line from car to projected point (shows e_y)
        self.draw_line(car.position, projected, color=(0, 255, 0), width=1)
    
    def draw_rays(self, car_pos, distances, hit_points, max_distance,
                  is_mirror=None):
        """
        Draw raycasting visualization on the track view.
        
        Forward rays: Red (close) -> Green (far)
        Mirror rays:  Magenta (close) -> Cyan (far)  — visually distinct
        
        Args:
            car_pos: (x, y) car world position
            distances: (N,) ray distances
            hit_points: (N, 2) ray hit positions
            max_distance: maximum ray range (for color scaling)
            is_mirror: (N,) bool mask — True for mirror rays
        """
        car_screen = self.world_to_screen(car_pos)
        
        for i in range(len(distances)):
            ratio = min(distances[i] / max_distance, 1.0)
            
            if is_mirror is not None and is_mirror[i]:
                # Mirror: magenta (close) -> cyan (far)
                r = int(200 * (1 - ratio))
                g = int(255 * ratio)
                b = 255
            else:
                # Forward: red (close) -> green (far)
                r = int(255 * (1 - ratio))
                g = int(255 * ratio)
                b = 0
            color = (r, g, b)
            
            hit_screen = self.world_to_screen(hit_points[i])
            pygame.draw.line(self.screen, color, car_screen, hit_screen, 1)
            pygame.draw.circle(self.screen, color, hit_screen, 3)
    
    def draw_sensor_panel(self, ray_distances, max_distance, is_mirror=None):
        """
        Draw sensor info panel below the lap timer (top-right area).
        Mini bar chart: forward bars in green tones, mirror bars in blue tones.
        
        Args:
            ray_distances: (N,) array of distances
            max_distance: max ray range
            is_mirror: (N,) bool mask
        """
        x = RENDER.screen_width - 260
        y = 200
        
        panel_w, panel_h = 260, 80
        panel_surface = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 0, 160))
        self.screen.blit(panel_surface, (x - 10, y - 5))
        
        num = len(ray_distances)
        n_mirror = int(np.sum(is_mirror)) if is_mirror is not None else 0
        label = f"RAYS {num-n_mirror}fwd+{n_mirror}mir  [V]"
        self._draw_text_small(label, (x, y), (180, 180, 180))
        y += 18
        
        bar_w = max(4, min(22, 240 // num))
        max_bar_h = 30
        
        for i, dist in enumerate(ray_distances):
            ratio = min(dist / max_distance, 1.0)
            bar_h = max(2, int(ratio * max_bar_h))
            
            if is_mirror is not None and is_mirror[i]:
                r = int(200 * (1 - ratio))
                g = int(255 * ratio)
                b = 255
            else:
                r = int(255 * (1 - ratio))
                g = int(255 * ratio)
                b = 0
            color = (r, g, b)
            
            bx = x + i * (bar_w + 2)
            by = y + max_bar_h - bar_h
            pygame.draw.rect(self.screen, color, (bx, by, bar_w, bar_h))
        
        y += max_bar_h + 4
        min_d = min(ray_distances)
        self._draw_text_small(f"Closest wall: {min_d:.1f}m", (x, y), (200, 200, 200))
    
    def draw_track_name(self, name, total_length):
        """Draw track name centered at top of screen."""
        text = f"{name}  ({total_length/1000:.1f} km)"
        surface = self.font.render(text, True, (200, 200, 200))
        x = (RENDER.screen_width - surface.get_width()) // 2
        self.screen.blit(surface, (x, 5))
        
        hint = self.font_small.render("[T] switch track", True, (100, 100, 100))
        hx = (RENDER.screen_width - hint.get_width()) // 2
        self.screen.blit(hint, (hx, 25))
    
    def draw_lap_timer(self, lap_data):
        """
        Draw lap timing panel in top-right corner (like real F1 timing).
        
        Args:
            lap_data: dict with keys:
                - current_time: current lap elapsed seconds
                - best_time: best lap time (inf if no lap completed)
                - lap_count: number of completed laps
                - wall_hits: total wall contacts this session
                - touching_wall: bool, currently in contact
                - current_sector: which sector car is in (0-indexed)
        """
        x = RENDER.screen_width - 260
        y = 10
        line_h = 22
        
        # Background panel
        panel_rect = pygame.Rect(x - 10, y - 5, 260, 182)
        panel_surface = pygame.Surface((260, 182), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 0, 160))
        self.screen.blit(panel_surface, (x - 10, y - 5))
        
        # Lap count
        lap_text = f"LAP {lap_data.get('lap_count', 0)}"
        self._draw_text_big(lap_text, (x, y), (255, 255, 255))
        y += line_h + 6
        
        # Current lap time
        cur_time = lap_data.get('current_time', 0.0)
        mins = int(cur_time // 60)
        secs = cur_time % 60
        time_str = f"TIME  {mins}:{secs:05.2f}"
        self._draw_text(time_str, (x, y), (255, 255, 255))
        y += line_h
        
        # Best lap time
        best = lap_data.get('best_time', float('inf'))
        if best < float('inf'):
            b_mins = int(best // 60)
            b_secs = best % 60
            best_str = f"BEST  {b_mins}:{b_secs:05.2f}"
            self._draw_text(best_str, (x, y), (180, 50, 255))
        else:
            self._draw_text("BEST  --:--.--", (x, y), (100, 100, 100))
        y += line_h
        
        # Current sector
        sector = lap_data.get('current_sector', 0) + 1
        sector_color = SECTOR_COLORS[(sector - 1) % len(SECTOR_COLORS)]
        self._draw_text(f"SECTOR {sector}", (x, y), sector_color)
        y += line_h
        
        # Wall hits
        hits = lap_data.get('wall_hits', 0)
        touching = lap_data.get('touching_wall', False)
        hit_color = (255, 50, 50) if touching else (200, 200, 200)
        hit_str = f"WALL HITS: {hits}"
        if touching:
            hit_str += "  !! CONTACT !!"
        self._draw_text(hit_str, (x, y), hit_color)
        y += line_h

        # Car hits
        car_hits = lap_data.get('car_hits', 0)
        hit_car_str = f"CAR HITS: {car_hits}"
        self._draw_text(hit_car_str, (x, y), (220, 120, 255))

    def draw_hud(self, car, frenet_data=None):
        """
        Draw heads-up display with car info (left side).
        
        Args:
            car: Car instance
            frenet_data: Optional Frenet coordinates to display
        """
        y_offset = 10
        line_height = 20
        
        # Speed
        speed_text = f"Speed: {car.speed:.1f} m/s ({car.speed * 3.6:.1f} km/h)"
        self._draw_text(speed_text, (10, y_offset))
        y_offset += line_height
        
        # Controls
        throttle_text = f"Throttle: {car.throttle:+.2f}  Steering: {car.steering:+.2f}"
        self._draw_text(throttle_text, (10, y_offset))
        y_offset += line_height
        
        # Frenet data if available
        if frenet_data:
            y_offset += 10
            self._draw_text("=== Frenet Frame ===", (10, y_offset))
            y_offset += line_height
            
            s_text = f"s (progress): {frenet_data['s']:.1f}m"
            self._draw_text(s_text, (10, y_offset))
            y_offset += line_height
            
            ey_text = f"e_y (lateral): {frenet_data['e_y']:.2f}m"
            self._draw_text(ey_text, (10, y_offset))
            y_offset += line_height
            
            epsi_text = f"e_psi (heading err): {np.degrees(frenet_data['e_psi']):.1f} deg"
            self._draw_text(epsi_text, (10, y_offset))
            y_offset += line_height
            
            kappa_text = f"kappa (curvature): {frenet_data['kappa']:.4f}"
            self._draw_text(kappa_text, (10, y_offset))
        
        # Instructions (bottom-left)
        instructions = [
            "Controls:",
            "  Up/W: Accelerate  Down/S: Brake",
            "  Left/A: Steer L   Right/D: Steer R",
            "  R: Reset  C: Camera  V: Sensors",
            "  T: Track  P: Screenshot  +/-: Zoom",
        ]
        
        for i, text in enumerate(instructions):
            self._draw_text_small(text, (10, RENDER.screen_height - 190 + i * 18),
                                  (160, 160, 160))
    
    def _draw_text(self, text, position, color=(255, 255, 255)):
        """Draw text at screen position (normal font)."""
        surface = self.font.render(text, True, color)
        self.screen.blit(surface, position)
    
    def _draw_text_big(self, text, position, color=(255, 255, 255)):
        """Draw text at screen position (big font)."""
        surface = self.font_big.render(text, True, color)
        self.screen.blit(surface, position)
    
    def _draw_text_small(self, text, position, color=(255, 255, 255)):
        """Draw text at screen position (small font)."""
        surface = self.font_small.render(text, True, color)
        self.screen.blit(surface, position)
    
    def update(self):
        """Update the display (call after all drawing)."""
        pygame.display.flip()
    
    def tick(self, fps=60):
        """
        Limit frame rate and return delta time.
        
        Args:
            fps: Target frames per second
            
        Returns:
            Time since last frame in seconds
        """
        return self.clock.tick(fps) / 1000.0
    
    def get_events(self):
        """Get Pygame events (for input handling)."""
        return pygame.event.get()
    
    def quit(self):
        """Clean up Pygame."""
        pygame.quit()

    def draw_cars(self, cars, collision_handler=None):
        """Draw all cars with per-car collision highlighting."""
        for car in cars:
            touching_wall = False
            touching_car = False
            if collision_handler is not None:
                st = collision_handler.get_car_stats(car.car_id)
                touching_wall = st['touching_wall']
                touching_car = st['touching_car']
            self.draw_car(car, touching_wall=touching_wall, touching_car=touching_car)
