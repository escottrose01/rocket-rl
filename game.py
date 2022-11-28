import pygame

from engine import GameObject, Game
from controller import RocketController, PlayerController, PIDController, OnOffController
from planet import PlaneGravitySource
from rocket import Rocket, NoisyRocket
from util import Circle

GROUND_Y = 550
GROUND_BOUNCE = 0.4
GROUND_FRICTION = 2.5
TARGET_Y = 200

GRAVITY = 9.8

ROCKET_MASS = 27670.
ROCKET_THRUST = 410000.
ROCKET_TORQUE = 100.

game = Game.instance()

all_sprites = pygame.sprite.Group()

target = (150, TARGET_Y)

rocket = Rocket((150, GROUND_Y), ROCKET_MASS, ROCKET_THRUST, ROCKET_TORQUE)
rocket2 = Rocket((100, GROUND_Y), ROCKET_MASS, ROCKET_THRUST, ROCKET_TORQUE)
# rocket = NoisyRocket((WIDTH/2, GROUND_Y), ROCKET_MASS, ROCKET_THRUST, ROCKET_TORQUE)
plane = PlaneGravitySource(GRAVITY, GROUND_Y, GROUND_BOUNCE, GROUND_FRICTION)

# controller = PIDController(rocket, target, 1.0, 0.0001, 2.3)
# controller = PIDController(rocket, target, 3.0, 0.0001, 2.3)
# controller = OnOffController(rocket, target)

controller = PlayerController(rocket)
game.register_callback(lambda _: controller.accept_input(pygame.key.get_pressed()))

controller2 = PIDController(rocket2, target, 1.0, 0.001, 2.3)

game.run()
