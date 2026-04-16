"""
Car Physics
A simplified car body for Phase 1.
Later we'll add proper tire physics.
"""
from Box2D import b2PolygonShape, b2Vec2
import numpy as np
import sys
sys.path.append('..')
from config import CAR


class Car:
    """
    A simple car represented as a single rectangular body.
    
    Phase 1: Single body with direct force application
    Later: Will have 4 tires with proper friction model
    """
    
    def __init__(self, world, position=(0, 0), angle=0):
        """
        Create a car in the physics world.
        
        Args:
            world: World instance (our wrapper)
            position: (x, y) starting position in meters
            angle: starting rotation in radians
        """
        self.world = world
        
        # Create the car body
        self.body = world.create_dynamic_body(
            position=position,
            angle=angle,
            linearDamping=CAR.drag_coefficient,
            angularDamping=4.0  # F1 cars have very tight yaw control
        )
        
        # Add rectangular shape (the car chassis)
        # Box2D uses half-widths for boxes
        self.body.CreatePolygonFixture(
            box=(CAR.length / 2, CAR.width / 2), #todo maybe change to 4.5 and 2.0 - understand why we devide by 2
            density=CAR.mass / (CAR.length * CAR.width),  # Mass distributed over area
            friction=0.3
        )
        
        # Tag body for collision identification
        self.body.userData = {'type': 'car'}
        
        # Control state
        self.throttle = 0.0    # -1 (reverse) to 1 (forward)
        self.steering = 0.0   # -1 (left) to 1 (right)
        
    @property
    def position(self):
        """Get car center position as numpy array"""
        pos = self.body.position
        return np.array([pos.x, pos.y])
    
    @property
    def angle(self):
        """Get car rotation in radians"""
        return self.body.angle
    
    @property
    def velocity(self):
        """Get linear velocity as numpy array"""
        vel = self.body.linearVelocity
        return np.array([vel.x, vel.y])
    
    @property
    def speed(self):
        """Get scalar speed (magnitude of velocity)"""
        return np.linalg.norm(self.velocity)
    
    @property
    def forward_vector(self):
        """Unit vector pointing in car's forward direction"""
        angle = self.angle
        return np.array([np.cos(angle), np.sin(angle)])
    
    @property
    def right_vector(self):
        """Unit vector pointing to car's right side"""
        angle = self.angle
        return np.array([np.cos(angle - np.pi/2), np.sin(angle - np.pi/2)])
    
    def get_forward_velocity(self):
        """Component of velocity in forward direction"""
        return np.dot(self.velocity, self.forward_vector)
    
    def get_lateral_velocity(self):
        """Component of velocity in sideways direction (drift)"""
        return np.dot(self.velocity, self.right_vector)
    
    def update(self):
        """
        Apply forces based on current control inputs.
        Call this every physics step.
        
        Key F1 physics modeled:
        - Aerodynamic downforce: grip increases with speed² (the F1 secret)
        - High braking force (~5G deceleration)
        - Steering authority stays strong at mid-high speeds
        """
        # === LATERAL FRICTION + AERODYNAMIC DOWNFORCE ===
        # Base grip from racing slicks + downforce bonus that grows with speed²
        # This is WHY F1 cars can corner at 250 km/h — more speed = more downforce = more grip
        lateral_vel = self.get_lateral_velocity()
        
        speed_ratio = min(self.speed / 90.0, 1.0)  # Normalize to ~top speed
        downforce_bonus = 0.07 * speed_ratio ** 2   # Up to +0.07 at top speed
        effective_grip = min(CAR.lateral_friction + downforce_bonus, 0.99)
        
        lateral_impulse = -lateral_vel * self.body.mass * effective_grip
        impulse_vec = self.right_vector * lateral_impulse
        self.body.ApplyLinearImpulse(
            b2Vec2(float(impulse_vec[0]), float(impulse_vec[1])),
            self.body.worldCenter,
            True
        )
        
        # === THROTTLE (forward/backward force) ===
        if self.throttle > 0:
            force_magnitude = self.throttle * CAR.max_forward_force
        else:
            force_magnitude = self.throttle * CAR.max_backward_force
        
        force_vec = self.forward_vector * force_magnitude
        self.body.ApplyForce(
            b2Vec2(float(force_vec[0]), float(force_vec[1])),
            self.body.worldCenter,
            True
        )
        
        # === STEERING (apply torque to rotate car) ===
        # F1 cars keep strong turning at mid-high speeds thanks to downforce
        # Uses gentler power curve so steering doesn't vanish at speed
        speed_factor = max(0.15, 1.0 - (self.speed / 120.0) ** 1.5)
        torque = self.steering * 45000 * speed_factor
        self.body.ApplyTorque(torque, True)
        
        # === ROLLING RESISTANCE ===
        if abs(self.throttle) < 0.1:
            forward_vel = self.get_forward_velocity()
            resistance = -forward_vel * self.body.mass * CAR.rolling_resistance
            resist_vec = self.forward_vector * resistance
            self.body.ApplyForce(
                b2Vec2(float(resist_vec[0]), float(resist_vec[1])),
                self.body.worldCenter,
                True
            )
    
    def set_controls(self, throttle, steering):
        """
        Set control inputs.
        
        Args:
            throttle: -1 (full reverse) to 1 (full forward)
            steering: -1 (full left) to 1 (full right)
        """
        self.throttle = np.clip(throttle, -1, 1)
        self.steering = np.clip(steering, -1, 1)


# Quick test
if __name__ == "__main__":
    from world import World
    
    world = World()
    car = Car(world, position=(0, 0), angle=0)
    
    print(f"Car created at: {car.position}")
    print(f"Car angle: {car.angle}")
    print(f"Forward vector: {car.forward_vector}")
    
    # Simulate throttle
    car.set_controls(throttle=1.0, steering=0)
    for i in range(60):  # 1 second at 60fps
        car.update()
        world.step()
    
    print(f"After 1 second of full throttle:")
    print(f"  Position: {car.position}")
    print(f"  Speed: {car.speed:.2f} m/s")

