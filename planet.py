"""A planetary RigidBody type."""

import numpy as np
import pygame

from physics import RigidBody, Physics


class GravitySource(RigidBody):
    """A gravitational body that can influence other RigidBodies"""

    G = 1e-5

    def __init__(self, mass: float, position: list):
        """Initializes a new GravitySource instance.

        Args:
            mass (float): the mass of this body.
            position (array_like): the initial (x, y) position of this body.
        """
        super().__init__(mass, np.Infinity, position, 0)

    def update(self):
        for b in Physics.instance().bodies:
            if b is not self:
                d = b.position - self._p
                r2 = np.inner(d, d)
                b.add_force(GravitySource.G * self.mass * b.mass / r2 * d)


class PlaneGravitySource(RigidBody):
    """A static plane with a gravitational pull."""

    def __init__(self, a, height):
        super().__init__(np.Infinity, np.Infinity, (0, height), 0)
        self._a = a
        self._height = height

        # pygame graphics
        self.sprite = PlaneGravitySource.Sprite(self._height)

    def update(self, dt):
        for b in Physics.instance().bodies:
            if b is not self:
                b.add_force(self._a * b.mass * np.array((0, 1), dtype=np.float64))
                if b.position[1] > self._height:
                    position = np.array((b.position[0], self._height), dtype=np.float64)
                    direction = np.array((0, -1), dtype=np.float64)
                    Physics.instance().add_collision((b, position, direction))

    def step(self, dt):
        pass

    class Sprite(pygame.sprite.Sprite):
        def __init__(self, height):
            super().__init__()
            self.image_ = pygame.image.load('res/ground.png').convert_alpha()
            self.rect_ = pygame.Rect(0, height, self.image_.get_width(), self.image_.get_height())
            self.rect = self.rect_
            self.image = self.image_