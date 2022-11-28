import numpy as np
import pygame

from physics import RigidBody


# forward declaration of classes
class TrajectoryInfo:
    pass


class Rocket(RigidBody):
    """A rocket equipped with a bottom thruster and reaction control systmes."""

    def __init__(self, position: list, mass: float, max_thrust: float, max_torque: float):
        """Initializes a new Rocket instance.

        Args:
            position (array_like): the initial (x, y) position of this rocket,
                specified from the bottom center.
            mass (float): the mass of this rocket.
            max_thrust (float): the maximum thrust force from this rocket's engine.
            max_torque (float): the maximum torque from this rocket's reaction control system.
            controller (Controller, optional): the Controller object used to fly
                this rocket. Defaults to None.
        """
        super().__init__(mass, 100., position, 0., 16, 16)
        self._max_thrust = max_thrust
        self._max_torque = max_torque
        self._thrust = 0.
        self._torque = 0.

        # pygame graphics
        self.rocket_sprite = Rocket.Sprite('res/lander.png', self.position[0], self.position[1])
        self.plume_sprite = Rocket.Sprite('res/plume.png', self.position[0], self.position[1])

    def update(self, dt):
        thrust_force = self._max_thrust * self._thrust * self.heading
        self.add_force(thrust_force)
        self.add_torque(self._torque * self._max_torque)

    def step(self, dt):
        super().step(dt)

        x = self.position[0]
        y = self.position[1]

        self.rocket_sprite.transform(x, y, self.rotation)

        x -= (self.rocket_sprite.image_.get_height()-4) * self._thrust * self.heading[0]
        y -= (self.rocket_sprite.image_.get_height()-4) * self._thrust * self.heading[1]

        self.plume_sprite.transform(x, y, self.rotation)

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
    def trajectory_info(self) -> TrajectoryInfo:
        """Returns a descriptive summary of the rocket's trajectory.

        Returns:
            TrajectoryInfo: the current trajectory of this rocket.
        """
        return TrajectoryInfo(self)

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


class TrajectoryInfo:
    """A description of a rocket's trajectory suitable as input to a control system"""

    def __init__(self, rocket):
        self.rocket = rocket
