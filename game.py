"""Simple 2D rocket simulator."""

import time
import pygame
from pygame.locals import (
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

from controller import RocketController, PlayerController, PIDController, OnOffController
from planet import PlaneGravitySource
from rocket import Rocket, NoisyRocket
from physics import Physics
from util import Circle

TITLE = 'RL-Rocket'
WIDTH = 800
HEIGHT = 600

FPS = 60
SCALE = 2

GROUND_Y = 550
GROUND_BOUNCE = 0.4
GROUND_FRICTION = 2.5
TARGET_Y = 200

GRAVITY = 9.8

ROCKET_MASS = 27670.
ROCKET_THRUST = 410000.
ROCKET_TORQUE = 100.

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

pygame.init()
screen = pygame.display.set_mode([WIDTH, HEIGHT])

all_sprites = pygame.sprite.Group()


class Game(object):
    """Manages the rocket game engine."""

    def __init__(self, rocket: Rocket, controller: RocketController = None):
        """Initializes a new Game instance.

        Args:
            rocket (Rocket): the rocket being flown.
            controller (RocketController, optional): the controller for the rocket. Defaults to None.

        """
        self._rocket = rocket
        self._controller = controller

    def run(self):
        """Runs the simulation until the user closes out."""

        clock = pygame.time.Clock()

        while True:
            # simulation update
            dt = SCALE * clock.tick(FPS) / 1000
            Physics.instance().step(dt)

            # graphics
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        return
                elif event.type == QUIT:
                    return

            screen.fill(BLACK)

            for entity in all_sprites:
                screen.blit(entity.image, entity.rect)

            if type(controller) == PlayerController:
                pressed_keys = pygame.key.get_pressed()
                controller.accept_input(pressed_keys)

            pygame.display.flip()


if __name__ == '__main__':
    rocket = Rocket((WIDTH/2, GROUND_Y), ROCKET_MASS, ROCKET_THRUST, ROCKET_TORQUE)
    # rocket = NoisyRocket((WIDTH/2, GROUND_Y), ROCKET_MASS, ROCKET_THRUST, ROCKET_TORQUE)

    plane = PlaneGravitySource(GRAVITY, GROUND_Y, GROUND_BOUNCE, GROUND_FRICTION)
    all_sprites.add(Circle(WIDTH//2, TARGET_Y, 5))
    all_sprites.add(rocket.plume_sprite)
    all_sprites.add(rocket.rocket_sprite)
    all_sprites.add(plane.sprite)

    target = (WIDTH//2 + 50, TARGET_Y)
    # controller = PlayerController(rocket)
    controller = PIDController(rocket, target, 1.0, 0.0001, 2.3)
    # controller = PIDController(rocket, target, 3.0, 0.0001, 2.3)
    # controller = OnOffController(rocket, target)

    Game(rocket, controller).run()
