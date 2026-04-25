"""
Box2D World Manager
Creates and manages the physics simulation world.
Includes collision detection via contact listener.
"""
from Box2D import b2World, b2Vec2, b2ContactListener
import sys
sys.path.append('..')
from config import SIM
from collections import defaultdict


class CollisionHandler(b2ContactListener):
    """
    Listens for Box2D contact events between bodies.
    
    When any two fixtures collide, Box2D calls BeginContact/EndContact.
    We check the body userData tags to identify WHAT collided:
      - 'car' + 'wall' = car hit a track barrier
    
    This data feeds into:
      - HUD display (wall hit indicator)
      - RL reward (penalty for crashing)
      - Stats tracking (total wall hits per lap)
    """
    
    def __init__(self):
        b2ContactListener.__init__(self)
        # Aggregate across all cars (telemetry only). For player UI use car_stats[car_id].
        self.touching_wall = False
        self._wall_contacts = 0
        self.total_wall_hits = 0
        self.wall_hit_speed = 0.0

        # new multi-car fields
        self.touching_car = False
        self._car_contacts = 0
        self.total_car_collisions = 0
        self.last_car_collision_speed = 0.0
        self.car_stats = defaultdict(lambda: {
            'touching_wall': False, 'touching_car': False,
            'wall_hit_count': 0, 'car_collision_count': 0,
            'wall_hit_speed': 0.0, 'car_collision_speed': 0.0,
            '_wall_contacts': 0, '_car_contacts': 0
        })
        self.step_count = 0
        self.ignore_car_collision_count_until_step = 0

    def _get_car_id(self, body):
        if isinstance(body.userData, dict):
            return body.userData.get('car_id', None)
        return None

    def BeginContact(self, contact):
        """Called by Box2D when two fixtures start overlapping."""
        a, b = contact.fixtureA.body, contact.fixtureB.body
        type_a, type_b = self._get_type(a), self._get_type(b)

        is_car_wall = (
            (type_a == 'car' and type_b == 'wall') or
            (type_a == 'wall' and type_b == 'car')
        )
        if is_car_wall:
            car_body = a if type_a == 'car' else b
            cid = self._get_car_id(car_body)
            allow_count = self.step_count >= self.ignore_car_collision_count_until_step

            if cid is not None:
                st = self.car_stats[cid]
                # always maintain touching state; count only after grace window
                if st['_wall_contacts'] == 0 and allow_count:
                    st['wall_hit_count'] += 1
                    st['wall_hit_speed'] = car_body.linearVelocity.length
                    self.total_wall_hits += 1
                    self.wall_hit_speed = car_body.linearVelocity.length
                st['_wall_contacts'] += 1
                st['touching_wall'] = True

            self.touching_wall = any(v['touching_wall'] for v in self.car_stats.values())

        if type_a == 'car' and type_b == 'car':
            rel_v = (a.linearVelocity - b.linearVelocity).length
            allow_count = self.step_count >= self.ignore_car_collision_count_until_step

            for body in (a, b):
                cid = self._get_car_id(body)
                if cid is None:
                    continue
                st = self.car_stats[cid]
                if st['_car_contacts'] == 0 and allow_count:
                    st['car_collision_count'] += 1
                    st['car_collision_speed'] = rel_v
                st['_car_contacts'] += 1
                st['touching_car'] = True

            if allow_count:
                self.total_car_collisions += 1
                self.last_car_collision_speed = rel_v
            self.touching_car = any(v['touching_car'] for v in self.car_stats.values())

    def EndContact(self, contact):
        """Called by Box2D when two fixtures stop overlapping."""
        a, b = contact.fixtureA.body, contact.fixtureB.body
        type_a, type_b = self._get_type(a), self._get_type(b)

        is_car_wall = (
            (type_a == 'car' and type_b == 'wall') or
            (type_a == 'wall' and type_b == 'car')
        )
        if is_car_wall:
            car_body = a if type_a == 'car' else b
            cid = self._get_car_id(car_body)
            if cid is not None:
                st = self.car_stats[cid]
                st['_wall_contacts'] = max(0, st['_wall_contacts'] - 1)
                st['touching_wall'] = st['_wall_contacts'] > 0

            self.touching_wall = any(v['touching_wall'] for v in self.car_stats.values())

        if type_a == 'car' and type_b == 'car':
            for body in (a, b):
                cid = self._get_car_id(body)
                if cid is None:
                    continue
                st = self.car_stats[cid]
                st['_car_contacts'] = max(0, st['_car_contacts'] - 1)
                st['touching_car'] = st['_car_contacts'] > 0

            self.touching_car = any(v['touching_car'] for v in self.car_stats.values())

    def get_car_stats(self, car_id):
        return self.car_stats[int(car_id)]

    def _get_type(self, body):
        """Safely extract the 'type' string from a body's userData dict."""
        if isinstance(body.userData, dict):
            return body.userData.get('type', '')
        return ''

    def reset(self):
        """Reset all counters (call on lap reset / episode reset)."""
        self.touching_wall = False
        self._wall_contacts = 0
        self.total_wall_hits = 0
        self.wall_hit_speed = 0.0
        self.touching_car = False
        self._car_contacts = 0
        self.total_car_collisions = 0
        self.last_car_collision_speed = 0.0
        self.car_stats.clear()
        self.step_count = 0
        self.ignore_car_collision_count_until_step = 0


class World:
    """
    Manages the Box2D physics world.
    
    For top-down car simulation, we use ZERO gravity.
    The car moves in 2D plane as if viewed from above.
    
    The collision_handler is automatically registered — any bodies
    tagged with userData={'type': 'car'} or {'type': 'wall'} will
    have their contacts tracked.
    """
    
    def __init__(self):
        # Zero gravity for top-down view
        # (gravity would pull car "down" through the screen!)
        self.world = b2World(gravity=(0, 0))
        
        # Track bodies in the world
        self.bodies = []
        
        # Collision detection — register with Box2D
        self.collision_handler = CollisionHandler()
        self.world.contactListener = self.collision_handler
        
    def step(self):
        """
        Advance physics simulation by one timestep.
        Call this every frame.
        """
        self.world.Step(
            SIM.time_step,
            SIM.velocity_iterations,
            SIM.position_iterations
        )
        # Clear forces after step (Box2D accumulates forces)
        self.world.ClearForces()
        self.collision_handler.step_count += 1

    def create_dynamic_body(self, position, angle=0, **kwargs):
        """
        Create a body that moves (like a car).
        
        Args:
            position: (x, y) in meters
            angle: rotation in radians
            **kwargs: additional body parameters
            
        Returns:
            Box2D body
        """
        body = self.world.CreateDynamicBody(
            position=position,
            angle=angle,
            **kwargs
        )
        self.bodies.append(body)
        return body
    
    def create_static_body(self, position=(0, 0), **kwargs):
        """
        Create a body that doesn't move (like track walls).
        
        Args:
            position: (x, y) in meters
            **kwargs: additional body parameters
            
        Returns:
            Box2D body
        """
        body = self.world.CreateStaticBody(
            position=position,
            **kwargs
        )
        self.bodies.append(body)
        return body
    
    def get_body_count(self):
        """Return number of bodies in the world"""
        return len(self.bodies)


# Quick test
if __name__ == "__main__":
    world = World()
    print(f"Created world with gravity: {world.world.gravity}")
    print(f"Bodies in world: {world.get_body_count()}")
    print(f"Collision handler active: {world.collision_handler is not None}")
    
    # Create a test body
    body = world.create_dynamic_body(position=(10, 10))
    print(f"Created body at: {body.position}")
    
    # Step the simulation
    world.step()
    print("Physics step completed!")
