"""
Sensors for AI Observation (Phase 5)

Two complementary sensor types for the RL agent:

1. RayCaster — LiDAR-like distance measurement to track walls.
   Casts rays outward from the car and measures how far away the boundaries are.
   Gives spatial awareness ("where are the walls?").

2. FrenetObserver — State-based observation using Frenet frame coordinates.
   Describes the car's pose relative to the track centerline plus upcoming curvature.
   Gives track-knowledge awareness ("where am I on the track, what's ahead?").

Both produce numpy arrays suitable for neural network input.
They can be used independently or concatenated for a richer observation.
"""
import numpy as np
from config import CAR


class RayCaster:
    """
    Casts rays from the car to measure distances to track boundaries.

    Layout (like a real F1 car's perception):
      - Forward arc: 24 rays spanning -90 to +90 deg (front semicircle)
      - Left mirror:  3 rays spanning ~135 to 165 deg  (rear-left)
      - Right mirror: 3 rays spanning ~-165 to -135 deg (rear-right)
      - Blind spot between 90-135 and directly behind — realistic!

    Total: 30 rays.  Order in the array: [right_mirror | forward | left_mirror]
    """

    def __init__(self, num_forward_rays=24, forward_spread=np.pi,
                 num_mirror_rays=3, mirror_start=np.radians(135),
                 mirror_end=np.radians(165), max_distance=100.0):
        self.max_distance = max_distance
        self.num_forward = num_forward_rays
        self.num_mirror_per_side = num_mirror_rays

        half = forward_spread / 2
        forward = np.linspace(-half, half, num_forward_rays)

        if num_mirror_rays > 0:
            left_mirror = np.linspace(mirror_start, mirror_end, num_mirror_rays)
            right_mirror = np.linspace(-mirror_end, -mirror_start, num_mirror_rays)
            self.ray_angles = np.concatenate([right_mirror, forward, left_mirror])
        else:
            self.ray_angles = forward

        self.num_rays = len(self.ray_angles)

        self.is_mirror = np.zeros(self.num_rays, dtype=bool)
        if num_mirror_rays > 0:
            self.is_mirror[:num_mirror_rays] = True
            self.is_mirror[-num_mirror_rays:] = True

    def _intersect_segments_with_rays(self, origin, dirs, seg_starts, seg_ends, distances, hit_points):
        """Update nearest hits for all rays against provided segments."""
        AB = seg_ends - seg_starts
        AO = origin - seg_starts
        for r in range(self.num_rays):
            d = dirs[r]
            denom = d[0] * AB[:, 1] - d[1] * AB[:, 0]
            valid = np.abs(denom) > 1e-10
            if not np.any(valid):
                continue

            AO_v = AO[valid]
            AB_v = AB[valid]
            denom_v = denom[valid]

            cross_ao_ab = AO_v[:, 0] * AB_v[:, 1] - AO_v[:, 1] * AB_v[:, 0]
            cross_ao_d = AO_v[:, 0] * d[1] - AO_v[:, 1] * d[0]

            t = -cross_ao_ab / denom_v
            u = -cross_ao_d / denom_v

            hits = (t > 0.01) & (u >= 0.0) & (u <= 1.0)
            if np.any(hits):
                min_t = np.min(t[hits])
                if min_t < distances[r]:
                    distances[r] = min_t
                    hit_points[r] = origin + min_t * d

    def _car_segments(self, cars, ego_car=None):
        """Convert car rectangles to segment lists in world coordinates."""
        starts, ends = [], []
        half_l = CAR.length / 2.0
        half_w = CAR.width / 2.0
        local = np.array([
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w]
        ], dtype=np.float64)

        for c in (cars or []):
            if ego_car is not None and c is ego_car:
                continue
            pos = np.asarray(c.position, dtype=np.float64)
            a = float(c.angle)
            rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float64)
            pts = (local @ rot.T) + pos
            nxt = np.roll(pts, -1, axis=0)
            starts.append(pts)
            ends.append(nxt)

        if not starts:
            return None, None
        return np.vstack(starts), np.vstack(ends)

    def cast(self, origin, heading, inner_boundary, outer_boundary, cars=None, ego_car=None):
        """
        Cast all rays and find nearest wall intersection for each.

        Uses vectorized ray-segment intersection testing against both
        inner and outer boundary polylines.

        Args:
            origin: (x, y) car position in world coordinates
            heading: Car heading in radians
            inner_boundary: Nx2 array of inner boundary points (closed loop)
            outer_boundary: Mx2 array of outer boundary points (closed loop)

        Returns:
            distances: (num_rays,) distances to nearest wall per ray
            hit_points: (num_rays, 2) world-space (x,y) of each hit
        """
        origin = np.asarray(origin, dtype=np.float64)
        abs_angles = self.ray_angles + heading
        dirs = np.column_stack([np.cos(abs_angles), np.sin(abs_angles)])

        distances = np.full(self.num_rays, self.max_distance)
        hit_points = np.zeros((self.num_rays, 2))
        for r in range(self.num_rays):
            hit_points[r] = origin + self.max_distance * dirs[r]

        # walls
        for boundary in [inner_boundary, outer_boundary]:
            A = boundary
            B = np.roll(boundary, -1, axis=0)
            self._intersect_segments_with_rays(origin, dirs, A, B, distances, hit_points)

        # cars as obstacles (optional)
        if cars is not None:
            A, B = self._car_segments(cars, ego_car=ego_car)
            if A is not None:
                self._intersect_segments_with_rays(origin, dirs, A, B, distances, hit_points)

        return distances, hit_points

    def get_normalized(self, distances):
        """Normalize distances to [0, 1] for neural network input."""
        return np.clip(distances / self.max_distance, 0.0, 1.0)


class FrenetObserver:
    """
    Computes Frenet-frame observation — the car's state relative to the track.

    Produces:
      [speed, e_y, e_psi, kappa, lookahead_curvature_1, ..., lookahead_curvature_N]

    This is the "driver who knows the track map" approach:
    - e_y: how far off-center am I?
    - e_psi: am I pointing the right way?
    - kappa: how tight is this turn?
    - lookahead: what turns are coming up?

    Like an F1 driver who has studied onboard laps and knows the
    track layout, but still needs to react to the car's current state.
    """

    def __init__(self, num_lookahead=10, lookahead_spacing=5.0):
        """
        Args:
            num_lookahead: Number of curvature lookahead points
            lookahead_spacing: Meters between lookahead samples
        """
        self.num_lookahead = num_lookahead
        self.lookahead_spacing = lookahead_spacing

    def observe(self, car, track):
        """
        Compute full Frenet-frame observation.

        Args:
            car: Car instance (needs .position, .angle, .speed)
            track: Track instance

        Returns:
            dict with:
                frenet: raw Frenet coordinate dict {s, e_y, e_psi, kappa, ...}
                lookahead: (N,) curvature lookahead array
                observation: flat numpy array for direct RL input
        """
        frenet = track.get_frenet_coordinates(car.position, car.angle)
        lookahead = track.get_lookahead_curvature(
            frenet['s'], self.num_lookahead, self.lookahead_spacing
        )

        obs = np.array([
            car.speed,
            frenet['e_y'],
            frenet['e_psi'],
            frenet['kappa'],
            *lookahead
        ], dtype=np.float32)

        return {
            'frenet': frenet,
            'lookahead': lookahead,
            'observation': obs,
        }

    @staticmethod
    def get_obs_dim(num_lookahead=10):
        """Total observation dimension: 4 base + N lookahead."""
        return 4 + num_lookahead
