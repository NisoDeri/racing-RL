"""
Track Definition and Geometry Functions
Implements Frenet frame calculations for state-based observation.

Key concepts:
- Centerline: A closed polyline defining the track's center
- s: Progress along the track (arc length from start)
- e_y: Lateral deviation from centerline (+ = right, - = left)
- e_psi: Heading error (difference between car heading and track tangent)
- kappa: Curvature at any point (1/radius, + = left turn, - = right turn)
"""
import numpy as np
from typing import Tuple, List, Optional
from Box2D import b2EdgeShape
import sys
sys.path.append('..')
from config import TRACK


class Track:
    """
    A racing track defined by a centerline polyline.
    
    Provides geometry calculations for:
    - Projecting points onto the centerline
    - Computing Frenet frame coordinates (s, e_y, e_psi)
    - Checking if points are within track boundaries
    - Computing curvature for look-ahead
    """
    
    def __init__(
        self,
        centerline_points: np.ndarray,
        width: float = None,
        name: str = "Custom Track",
        generation_seed: int = None,
        generation_parameters: dict = None,
        verbose: bool = False,
    ):
        """
        Create a track from centerline points.
        
        Args:
            centerline_points: Nx2 array of (x, y) points defining centerline
                              Should form a closed loop (last connects to first)
            width: Track width in meters (uses config default if None)
            name: Human-readable track identifier for metrics and rendering.
            generation_seed: Procedural seed, or None for hand-authored tracks.
            generation_parameters: Optional procedural coefficients for reporting.
            verbose: Print boundary/wall construction diagnostics.
        """
        self.centerline = np.array(centerline_points, dtype=np.float64)
        self.width = width if width is not None else TRACK.width
        self.half_width = self.width / 2
        self.name = name
        self.generation_seed = generation_seed
        self.generation_parameters = generation_parameters or {}
        self.verbose = verbose
        
        # Boundary cache (computed lazily by get_boundary_points)
        self._cached_inner = None
        self._cached_outer = None

        # Precompute segment data for efficiency
        self._compute_segments()
        
    def _compute_segments(self):
        """Precompute segment vectors, lengths, and cumulative arc length."""
        n = len(self.centerline)
        
        # Segment vectors (from point i to point i+1, wrapping around)
        self.segments = np.zeros((n, 2))
        for i in range(n):
            next_i = (i + 1) % n
            self.segments[i] = self.centerline[next_i] - self.centerline[i]
        
        # Segment lengths
        self.segment_lengths = np.linalg.norm(self.segments, axis=1)
        
        # Cumulative arc length (s value at each point)
        self.cumulative_length = np.zeros(n + 1)
        for i in range(n):
            self.cumulative_length[i + 1] = self.cumulative_length[i] + self.segment_lengths[i]
        
        # Total track length
        self.total_length = self.cumulative_length[-1]
        
        # Unit tangent vectors at each segment
        self.tangents = self.segments / (self.segment_lengths[:, np.newaxis] + 1e-10)
        
        # Normal vectors (perpendicular to tangent, pointing "right")
        # Rotate tangent 90° clockwise
        self.normals = np.zeros_like(self.tangents)
        self.normals[:, 0] = self.tangents[:, 1]   # x = sin
        self.normals[:, 1] = -self.tangents[:, 0]  # y = -cos
        
        # Precompute curvature at each point
        self._compute_curvature()
    
    def _compute_curvature(self):
        """
        Compute curvature (kappa) at each centerline point.
        Curvature = rate of change of heading angle per unit length.
        Positive = turning left, Negative = turning right.
        """
        n = len(self.centerline)
        self.curvature = np.zeros(n)
        
        for i in range(n):
            # Get tangent angles of adjacent segments
            prev_i = (i - 1) % n
            
            angle_prev = np.arctan2(self.tangents[prev_i, 1], self.tangents[prev_i, 0])
            angle_curr = np.arctan2(self.tangents[i, 1], self.tangents[i, 0])
            
            # Angle change (handle wraparound)
            delta_angle = angle_curr - angle_prev
            if delta_angle > np.pi:
                delta_angle -= 2 * np.pi
            elif delta_angle < -np.pi:
                delta_angle += 2 * np.pi
            
            # Arc length over which this change occurs
            arc_length = (self.segment_lengths[prev_i] + self.segment_lengths[i]) / 2
            
            # Curvature = d(angle) / ds
            if arc_length > 1e-6:
                self.curvature[i] = delta_angle / arc_length
            else:
                self.curvature[i] = 0
    
    def project_point(self, point: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Project a point onto the centerline.
        
        Args:
            point: (x, y) position to project
            
        Returns:
            segment_idx: Index of the closest segment
            t: Parameter along segment (0 = start, 1 = end)
            projected: The projected point on the centerline
        """
        point = np.array(point)
        min_dist_sq = float('inf')
        best_segment = 0
        best_t = 0
        best_projected = self.centerline[0]
        
        for i in range(len(self.centerline)):
            # Vector from segment start to point
            to_point = point - self.centerline[i]
            
            # Project onto segment
            seg_len_sq = self.segment_lengths[i] ** 2
            if seg_len_sq < 1e-10:
                t = 0
            else:
                t = np.dot(to_point, self.segments[i]) / seg_len_sq
                t = np.clip(t, 0, 1)
            
            # Projected point
            projected = self.centerline[i] + t * self.segments[i]
            
            # Distance squared
            dist_sq = np.sum((point - projected) ** 2)
            
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_segment = i
                best_t = t
                best_projected = projected
        
        return best_segment, best_t, best_projected
    
    def get_frenet_coordinates(self, position: np.ndarray, heading: float) -> dict:
        """
        Get Frenet frame coordinates for a point with heading.
        This is the CORE function for state-based observation!
        
        Args:
            position: (x, y) position of the car
            heading: Car heading angle in radians
            
        Returns:
            Dictionary with:
            - s: Progress along track (0 to total_length)
            - e_y: Lateral deviation (+ = right of center, - = left)
            - e_psi: Heading error (+ = pointing right of track, - = left)
            - kappa: Curvature at this point
            - projected: The projected point on centerline
            - segment_idx: Which segment we're on
        """
        # Project onto centerline
        seg_idx, t, projected = self.project_point(position)
        
        # === Calculate s (progress) ===
        s = self.cumulative_length[seg_idx] + t * self.segment_lengths[seg_idx]
        
        # === Calculate e_y (lateral deviation) ===
        to_car = position - projected
        # Positive if car is to the RIGHT of centerline
        e_y = np.dot(to_car, self.normals[seg_idx])
        
        # === Calculate e_psi (heading error) ===
        track_heading = np.arctan2(self.tangents[seg_idx, 1], self.tangents[seg_idx, 0])
        e_psi = heading - track_heading
        # Normalize to [-pi, pi]
        while e_psi > np.pi:
            e_psi -= 2 * np.pi
        while e_psi < -np.pi:
            e_psi += 2 * np.pi
        
        # === Get curvature at this point ===
        # Interpolate between segment start and end curvature
        next_idx = (seg_idx + 1) % len(self.centerline)
        kappa = (1 - t) * self.curvature[seg_idx] + t * self.curvature[next_idx]
        
        return {
            's': s,
            'e_y': e_y,
            'e_psi': e_psi,
            'kappa': kappa,
            'projected': projected,
            'segment_idx': seg_idx
        }
    
    def get_lookahead_curvature(self, s: float, num_samples: int = None, 
                                 sample_distance: float = None) -> np.ndarray:
        """
        Get curvature values at points ahead of current position.
        This gives the AI "knowledge" of upcoming turns!
        
        Args:
            s: Current progress along track
            num_samples: Number of look-ahead points (default from config)
            sample_distance: Distance between samples in meters
            
        Returns:
            Array of curvature values at each look-ahead point
        """
        if num_samples is None:
            num_samples = TRACK.num_curvature_samples
        if sample_distance is None:
            sample_distance = TRACK.curvature_sample_distance
        
        curvatures = np.zeros(num_samples)
        
        for i in range(num_samples):
            # s value of look-ahead point
            lookahead_s = (s + (i + 1) * sample_distance) % self.total_length
            
            # Find which segment this s falls in
            seg_idx = np.searchsorted(self.cumulative_length[1:], lookahead_s)
            seg_idx = min(seg_idx, len(self.centerline) - 1)
            
            # Interpolation parameter within segment
            s_in_seg = lookahead_s - self.cumulative_length[seg_idx]
            t = s_in_seg / (self.segment_lengths[seg_idx] + 1e-10)
            t = np.clip(t, 0, 1)
            
            # Interpolate curvature
            next_idx = (seg_idx + 1) % len(self.centerline)
            curvatures[i] = (1 - t) * self.curvature[seg_idx] + t * self.curvature[next_idx]
        
        return curvatures
    
    def is_inside_track(self, position: np.ndarray) -> bool:
        """
        Check if a position is within track boundaries.

        Args:
            position: (x, y) position to check

        Returns:
            True if inside track, False if outside
        """
        _, _, projected = self.project_point(position)
        distance = np.linalg.norm(position - projected)
        return distance <= self.half_width
    
    def get_boundary_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the inner and outer boundary lines of the track.
        Uses Shapely polygon buffer for self-intersection-free offset curves.
        Results are cached since track geometry is static.
        
        Returns:
            inner_boundary: Nx2 array of inner edge points
            outer_boundary: Mx2 array of outer edge points
        """
        if self._cached_inner is not None:
            return self._cached_inner, self._cached_outer
        
        try:
            inner, outer = self._compute_boundary_shapely()
            if self.verbose:
                print(
                    f"  Boundary (Shapely): inner={len(inner)} pts, "
                    f"outer={len(outer)} pts"
                )
        except Exception as e:
            if self.verbose:
                print(f"  Shapely boundary failed ({e}), using naive offset")
            inner, outer = self._compute_boundary_naive()
        
        self._cached_inner = inner
        self._cached_outer = outer
        return inner, outer
    
    def _compute_boundary_shapely(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute clean offset boundaries using Shapely polygon buffer.
        
        Shapely's buffer operation uses the Minkowski sum algorithm which
        automatically handles self-intersections that occur when the offset
        distance exceeds the local radius of curvature at tight turns.
        
        This is the definitive fix for the "triangle artifact" at sharp corners.
        """
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.validation import make_valid

        poly = Polygon(self.centerline)
        if not poly.is_valid:
            poly = make_valid(poly)

        outer_poly = poly.buffer(self.half_width, quad_segs=8, join_style=1)
        inner_poly = poly.buffer(-self.half_width, quad_segs=8, join_style=1)

        if inner_poly.is_empty:
            raise ValueError("Inner buffer empty — half_width exceeds min turn radius everywhere")

        if isinstance(inner_poly, MultiPolygon):
            inner_poly = max(inner_poly.geoms, key=lambda g: g.area)
        if isinstance(outer_poly, MultiPolygon):
            outer_poly = max(outer_poly.geoms, key=lambda g: g.area)

        outer = np.array(outer_poly.exterior.coords[:-1])
        inner = np.array(inner_poly.exterior.coords[:-1])

        # Align starting points: Shapely picks arbitrary start positions
        # for each ring. If they differ, the rendering polygon
        # (outer + inner[::-1]) gets a visible diagonal seam.
        # Fix: rotate inner so its start is closest to outer's start.
        dists = np.linalg.norm(inner - outer[0], axis=1)
        best = np.argmin(dists)
        if best != 0:
            inner = np.roll(inner, -best, axis=0)

        return inner, outer
    
    def _compute_boundary_naive(self) -> Tuple[np.ndarray, np.ndarray]:
        """Original per-point offset (fallback if Shapely unavailable)."""
        n = len(self.centerline)
        inner = np.zeros((n, 2))
        outer = np.zeros((n, 2))
        
        for i in range(n):
            prev_i = (i - 1) % n
            avg_normal = (self.normals[prev_i] + self.normals[i]) / 2
            avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-10)
            
            inner[i] = self.centerline[i] - avg_normal * self.half_width
            outer[i] = self.centerline[i] + avg_normal * self.half_width
        
        return inner, outer
    
    def create_walls(self, world):
        """
        Create Box2D static edge bodies for BOTH track boundaries.
        
        Each boundary (inner + outer) becomes one static body with many
        edge fixtures — one edge per segment of the boundary polyline.
        
        Box2D then handles car-wall collision automatically:
        - Car bounces off (controlled by wall_restitution)
        - Car loses speed scraping (controlled by wall_friction)
        
        Args:
            world: World instance (our physics wrapper)
            
        Returns:
            (inner_wall_body, outer_wall_body) — the two Box2D static bodies
        """
        inner, outer = self.get_boundary_points()
        n_inner = len(inner)
        n_outer = len(outer)

        # --- Inner wall ---
        inner_wall = world.create_static_body()
        inner_wall.userData = {'type': 'wall', 'side': 'inner'}
        for i in range(n_inner):
            j = (i + 1) % n_inner
            inner_wall.CreateFixture(
                shape=b2EdgeShape(vertices=[
                    (float(inner[i][0]), float(inner[i][1])),
                    (float(inner[j][0]), float(inner[j][1]))
                ]),
                friction=TRACK.wall_friction,
                restitution=TRACK.wall_restitution,
            )

        # --- Outer wall ---
        outer_wall = world.create_static_body()
        outer_wall.userData = {'type': 'wall', 'side': 'outer'}
        for i in range(n_outer):
            j = (i + 1) % n_outer
            outer_wall.CreateFixture(
                shape=b2EdgeShape(vertices=[
                    (float(outer[i][0]), float(outer[i][1])),
                    (float(outer[j][0]), float(outer[j][1]))
                ]),
                friction=TRACK.wall_friction,
                restitution=TRACK.wall_restitution,
            )

        self.wall_bodies = [inner_wall, outer_wall]
        if self.verbose:
            print(
                f"  Track walls created: {n_inner} inner + {n_outer} outer "
                f"= {n_inner + n_outer} edges"
            )
        return inner_wall, outer_wall
    
    def get_checkpoint_positions(self, num_checkpoints=None):
        """
        Calculate checkpoint/sector line positions around the track.
        
        Returns a list of dicts, each with:
          - index: checkpoint number
          - s: arc-length position on centerline
          - center: (x,y) centerline point
          - inner: (x,y) inner boundary point (line endpoint)
          - outer: (x,y) outer boundary point (line endpoint)
          - is_start_finish: True for checkpoint 0
          
        These are used for:
          - Rendering sector/start-finish lines
          - Lap timing (detect when car crosses s=0)
          - Sector timing (split lap into segments)
        
        Args:
            num_checkpoints: how many checkpoints (default: config num_sectors)
            
        Returns:
            List of checkpoint dicts
        """
        if num_checkpoints is None:
            num_checkpoints = TRACK.num_sectors
        
        checkpoints = []
        for i in range(num_checkpoints):
            s = (i / num_checkpoints) * self.total_length
            
            # Find which segment this s falls in
            seg_idx = np.searchsorted(self.cumulative_length[1:], s)
            seg_idx = min(seg_idx, len(self.centerline) - 1)
            
            # Interpolation parameter within the segment
            s_in_seg = s - self.cumulative_length[seg_idx]
            t = s_in_seg / (self.segment_lengths[seg_idx] + 1e-10)
            t = np.clip(t, 0, 1)
            
            # Interpolate centerline position
            next_idx = (seg_idx + 1) % len(self.centerline)
            center = (1 - t) * self.centerline[seg_idx] + t * self.centerline[next_idx]
            
            # Get normal at this point (average of adjacent segment normals)
            normal = self.normals[seg_idx]
            
            # Boundary endpoints of the checkpoint line
            inner_pt = center - normal * self.half_width
            outer_pt = center + normal * self.half_width
            
            checkpoints.append({
                'index': i,
                's': s,
                'center': center,
                'inner': inner_pt,
                'outer': outer_pt,
                'is_start_finish': (i == 0)
            })
        
        return checkpoints
    
    @staticmethod
    def create_oval_track(center=(0, 0), length=200, width_radius=50, 
                          num_points=60, track_width=12) -> 'Track':
        """
        Create a simple oval track for testing.
        
        Args:
            center: Center position of the oval
            length: Total length of the straight sections
            width_radius: Radius of the curved ends
            num_points: Number of points in the centerline
            track_width: Width of the track
            
        Returns:
            Track instance
        """
        points = []
        cx, cy = center
        
        # Create oval shape
        # Two straights + two semicircles
        half_straight = length / 2
        
        for i in range(num_points):
            t = i / num_points * 2 * np.pi
            
            # Parametric oval
            x = (half_straight + width_radius) * np.cos(t)
            y = width_radius * np.sin(t)
            
            points.append([cx + x, cy + y])
        
        return Track(np.array(points), width=track_width, name="Oval")
    
    @staticmethod
    def create_sprint_track(track_width=12) -> 'Track':
        """
        Compact sprint circuit (~750m).
        Same Fourier harmonic approach, smaller base radius.
        Good for quick testing and tight racing.
        """
        num_points = 200
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        
        radius = (100
                  + 30 * np.cos(2 * angles)
                  + 15 * np.sin(3 * angles)
                  + 8 * np.cos(5 * angles)
                  + 5 * np.sin(7 * angles))
        
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        return Track(
            np.column_stack([x, y]), width=track_width, name="Sprint Circuit"
        )
    
    @staticmethod
    def create_complex_track(track_width=12) -> 'Track':
        """
        Full-length Grand Prix circuit (~3.5 km).
        
        Features:
        - Two long straights (cos 2θ)
        - Varied turn radii (sin 3θ asymmetry)
        - Tight hairpin sections (cos 5θ)
        - Chicane bumps (sin 7θ)
        - Esses section: 2-3 rapid hard turns in sequence,
          like Maggots-Becketts at Silverstone or Suzuka S-curves.
          Built with a Gaussian-windowed high-frequency oscillation.
        """
        num_points = 500
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        
        radius = (480
                  + 140 * np.cos(2 * angles)
                  + 65 * np.sin(3 * angles)
                  + 55 * np.cos(5 * angles)
                  + 30 * np.sin(7 * angles))
        
        # Esses section: localized cluster of 2-3 hard turns in sequence.
        # Gaussian window confines the oscillation to one part of the track,
        # so the rest keeps its original character.
        esses_window = np.exp(-0.5 * ((angles - 1.2) / 0.35) ** 2)
        radius += 45 * np.sin(14 * angles) * esses_window
        
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        return Track(
            np.column_stack([x, y]), width=track_width, name="Grand Prix Circuit"
        )

    def get_pose_at_s(self, s: float):
        """Return (position, heading, segment_idx) on centerline at arc-length s."""
        s_wrapped = s % self.total_length
        seg_idx = np.searchsorted(self.cumulative_length[1:], s_wrapped)
        seg_idx = min(seg_idx, len(self.centerline) - 1)
        s_in_seg = s_wrapped - self.cumulative_length[seg_idx]
        t = s_in_seg / (self.segment_lengths[seg_idx] + 1e-10)
        t = np.clip(t, 0.0, 1.0)
        next_idx = (seg_idx + 1) % len(self.centerline)
        pos = (1 - t) * self.centerline[seg_idx] + t * self.centerline[next_idx]
        heading = np.arctan2(self.tangents[seg_idx, 1], self.tangents[seg_idx, 0])
        return pos, heading, seg_idx


# Quick test
if __name__ == "__main__":
    # Create a simple oval track
    track = Track.create_oval_track()
    print(f"Track created with {len(track.centerline)} points")
    print(f"Total length: {track.total_length:.1f} meters")
    print(f"Track width: {track.width} meters")
    
    # Test projection
    test_point = np.array([50, 5])
    frenet = track.get_frenet_coordinates(test_point, heading=0)
    print(f"\nTest point {test_point}:")
    print(f"  s (progress): {frenet['s']:.1f}m")
    print(f"  e_y (lateral): {frenet['e_y']:.2f}m")
    print(f"  e_psi (heading error): {np.degrees(frenet['e_psi']):.1f}°")
    print(f"  Inside track: {track.is_inside_track(test_point)}")
    
    # Test look-ahead curvature
    curvatures = track.get_lookahead_curvature(frenet['s'])
    print(f"\nLook-ahead curvatures: {curvatures}")
