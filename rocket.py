import numpy as np
import pygame

from engine import RigidBody, Collision


# forward declaration of classes
class TrajectoryInfo:
    pass


class Rocket(RigidBody):
    """A rocket equipped with a bottom thruster and reaction control systmes."""

    def __init__(self, position: list, mass: float, max_thrust: float, max_torque: float, rocket_sprite: str = 'res/lander.png', plume_sprite: str = 'res/plume.png'):
        """Initializes a new Rocket instance.

        Args:
            position (array_like): the initial (x, y) position of this rocket,
                specified from the bottom center.
            mass (float): the mass of this rocket.
            max_thrust (float): the maximum thrust force from this rocket's engine.
            max_torque (float): the maximum torque from this rocket's reaction control system.
            controller (Controller, optional): the Controller object used to fly
                this rocket. Defaults to None.
            rocket_sprite (str, optional): The location of the sprite for the rocket. Defaults to 'res/lander.png'.
            plume_sprite (str, optional): The location of the sprite for the plume. Defaults to 'res/plume.png'.
        """
        super().__init__(mass, 100., position, 0., 16, 16)
        self._max_thrust = max_thrust
        self._max_torque = max_torque
        self._thrust = 0.
        self._torque = 0.
        self.crashed = False
        self.grounded = False

        # pygame graphics
        self._rocket_sprite = Rocket.Sprite(rocket_sprite, self.position[0], self.position[1])
        self._plume_sprite = Rocket.Sprite(plume_sprite, self.position[0], self.position[1])

    def update(self, dt):
        self.grounded = False
        thrust_force = self._max_thrust * self._thrust * self.heading
        self.add_force(thrust_force)
        self.add_torque(self._torque * self._max_torque)

    def step(self, dt):
        super().step(dt)

        x = self.position[0]
        y = self.position[1]

        self._rocket_sprite.transform(x, y, self.rotation)

        x -= (self._rocket_sprite.image_.get_height()-2) * self._thrust * self.heading[0]
        y -= (self._rocket_sprite.image_.get_height()-2) * self._thrust * self.heading[1]

        self._plume_sprite.transform(x, y, self.rotation)

    @property
    def position_obs(self) -> np.ndarray:
        """Returns a noisy observation of this rocket's position.

        Returns:
            np.ndarray: the observation of this rocket's position.
        """
        return super().position

    @property
    def rotation_obs(self) -> float:
        """Returns a noisy observtion of this rocket's rotation.

        Returns:
            float: the observation of the rotation of this rocket.
        """
        return self.rotation

    @property
    def velocity_obs(self) -> np.ndarray:
        """Returns a noisy observation of this rocket's velocity.

        Returns:
            np.ndarray: the observation of this rocket's velocity.
        """
        return super().velocity

    @property
    def thrust(self) -> float:
        """Returns the current thrust of this rocket.

        Returns:
            float: the thrust of this rocket, in the range [0, 1].
        """
        return self._thrust

    @thrust.setter
    def thrust(self, value):
        """Sets the current thrust of this rocket.

        Args:
            value (float): the thrust of this rocket, as a percentage of its maximum thrust.
        """
        self._thrust = min(max(value, 0.0), 1.0)

    @property
    def torque(self) -> float:
        """Returns the current torque of this rocket.

        Returns:
            float: the thrust of this rocket, as a signed percentage of its maximum torque.
        """
        return self._torque

    @torque.setter
    def torque(self, value):
        """Sets the current torque of this rocket.

        Args:
            value (float): the torque of this rocket, as a signed percentage of its maximum torque.
        """
        self._torque = min(max(value, -1.0), 1.0)

    @property
    def max_thrust(self) -> float:
        """Returns the maximum thrust force of this rocket.

        Returns:
            float: the maximum thrust of this rocket.
        """
        return self._max_thrust

    @property
    def max_torque(self) -> float:
        """Returns the maximum torque of this rocket.

        Returns:
            float: the maximum torque of this rocket.
        """
        return self._max_torque

    @property
    def heading(self) -> np.ndarray:
        """Returns the heading of this rocket, equal to (sin(rotation), -cos(rotation))

        Returns:
            np.ndarray: the heading of this rocket.
        """
        return np.array([np.sin(self.rotation), -np.cos(self.rotation)])

    @property
    def heading_obs(self) -> np.ndarray:
        """Returns a noisy observation of the heading of this rocket, equal to (sin(rotation), -cos(rotation))

        Returns:
            np.ndarray: the observation of the heading of this rocket.
        """
        return np.array([np.sin(self.rotation), -np.cos(self.rotation)])

    @property
    def trajectory_info(self) -> TrajectoryInfo:
        """Returns a descriptive summary of the rocket's trajectory.

        Returns:
            TrajectoryInfo: the current trajectory of this rocket.
        """
        return TrajectoryInfo(self)

    def on_collision(self, collision: Collision, dt: float):
        self.grounded = True
        if np.linalg.norm(self.velocity) > 25 or np.cos(self.rotation) < 0:
            self.crashed = True

        self.position = collision.position
        v_n = max(collision.t, -collision.bounce * collision.t) * collision.normal
        v_t = self.velocity - collision.t*collision.normal
        s = np.linalg.norm(v_t)
        if (s != 0):
            v_t *= max(0, s - collision.friction * dt) / s
        self.velocity = v_n + v_t

        self.angular_velocity = max(0.0, abs(self.angular_velocity) -
                                    collision.friction/10 * dt) * np.sign(self.angular_velocity)

    def get_sprites(self) -> pygame.sprite.Sprite:
        return self._plume_sprite, self._rocket_sprite

    class Sprite(pygame.sprite.Sprite):
        """A simple subclass to facilitate graphics."""

        def __init__(self, fname, x, y):
            super().__init__()
            self.image_ = pygame.image.load(fname).convert_alpha()
            self.rect_ = pygame.Rect(0, 0, self.image_.get_width(), self.image_.get_height())
            self.rect_.midbottom = (x, y)
            self.rect = self.rect_
            self.image = self.image_

        def transform(self, x, y, angle):
            """Transform this sprite according to a given position and rotation.

            Args:
                x (int): the x position of the transformed sprite, in screen coordinates.
                y (int): the y position of the transformed sprite, in screen coordinates.
                angle (float): the orientation of the transformed sprite, in radians.
            """
            self.rect_ = pygame.Rect(0, 0, self.image_.get_width(), self.image_.get_height())
            self.rect_.midbottom = (x, y)
            self.image = pygame.transform.rotate(self.image_, -np.degrees(angle))
            self.rect = self.image.get_rect(center=self.rect_.center)


class NoisyRocket(Rocket):
    """A rocket which produces noisy inputs and outputs."""

    @property
    def position_obs(self) -> np.ndarray:
        return super().position + np.random.normal(0, 10)

    @property
    def velocity_obs(self) -> np.ndarray:
        return super().velocity + np.random.normal(0, 2)

    @property
    def rotation_obs(self) -> float:
        return super().rotation + np.random.normal(0, 0.03)

    @property
    def heading_obs(self) -> float:
        rot = super().rotation + np.random.normal(0, 0.03)
        return np.array([np.sin(rot), -np.cos(rot)])

    # def update(self, dt):
    #     noise = np.random.normal()/5.0
    #     thrust = self._thrust + noise
    #     thrust = min(1.0, max(0.0, thrust))
    #     thrust_force = self._max_thrust * thrust * self.heading
    #     self.add_force(thrust_force)
    #     self.add_torque(self._torque * self._max_torque)


class TrajectoryInfo:
    """A description of a rocket's trajectory suitable as input to a control system"""

    def __init__(self, rocket):
        self.rocket = rocket
