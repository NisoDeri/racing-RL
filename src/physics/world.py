"""
Box2D World Manager
Creates and manages the physics simulation world.
Includes collision detection via contact listener.
"""
from Box2D import b2World, b2Vec2, b2ContactListener
import sys
sys.path.append('..')
from config import SIM


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
        self.touching_wall = False
        self._wall_contacts = 0       # Active contact count (can touch multiple edges)
        self.total_wall_hits = 0      # Cumulative hits this session
        self.wall_hit_speed = 0.0     # Speed at moment of last impact (m/s)
    
    def BeginContact(self, contact):
        """Called by Box2D when two fixtures start overlapping."""
        type_a = self._get_type(contact.fixtureA.body)
        type_b = self._get_type(contact.fixtureB.body)
        
        if 'car' in (type_a, type_b) and 'wall' in (type_a, type_b):
            if self._wall_contacts == 0:
                # First contact edge — record as a new hit
                self.total_wall_hits += 1
                car_body = contact.fixtureA.body if type_a == 'car' else contact.fixtureB.body
                self.wall_hit_speed = car_body.linearVelocity.length
            self._wall_contacts += 1
            self.touching_wall = True
    
    def EndContact(self, contact):
        """Called by Box2D when two fixtures stop overlapping."""
        type_a = self._get_type(contact.fixtureA.body)
        type_b = self._get_type(contact.fixtureB.body)
        
        if 'car' in (type_a, type_b) and 'wall' in (type_a, type_b):
            self._wall_contacts = max(0, self._wall_contacts - 1)
            if self._wall_contacts == 0:
                self.touching_wall = False
    
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
