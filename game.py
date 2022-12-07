import pygame
import numpy as np

from engine import Game, HEIGHT
from controller import PlayerController, PIDController, DQNController, OnOffController
from planet import PlaneGravitySource
from rocket import Rocket, NoisyRocket
from util import Circle
from camera import KeyboardCamera

GROUND_Y = 0
GROUND_BOUNCE = 0.4
GROUND_FRICTION = 2.5
TARGET_Y = 400

GRAVITY = 9.8

ROCKET_MASS = 27670.
ROCKET_THRUST = 410000.
ROCKET_TORQUE = 20.

game = Game.instance()
# camera = KeyboardCamera(width=400, height=300)
camera = KeyboardCamera(y=275)

all_sprites = pygame.sprite.Group()

target = (400, TARGET_Y)


plane = PlaneGravitySource(GRAVITY, GROUND_Y, GROUND_BOUNCE, GROUND_FRICTION)
Circle(*target, 5)

# myrocket = Rocket((600, GROUND_Y), ROCKET_MASS, ROCKET_THRUST, ROCKET_TORQUE)
# mycontroller = PlayerController(myrocket)
# game.register_callback(lambda _, __: mycontroller.accept_input(pygame.key.get_pressed()))
# game.register_callback(lambda *_: camera.accept_input(pygame.key.get_pressed()))

# dumb_rocket = Rocket((400, GROUND_Y), ROCKET_MASS, ROCKET_THRUST, ROCKET_TORQUE)
# dumb_controller = OnOffController(dumb_rocket, target)

# for i in range(1, 13):
#     rocket = Rocket((100 + i*50, GROUND_Y), ROCKET_MASS, ROCKET_THRUST, ROCKET_TORQUE)
#     kp = 1.0
#     ki = 0.001
#     kd = 0.3 * i
#     controller = PIDController(rocket, target, kp, ki, kd)

dqn_rocket = Rocket((650, GROUND_Y), ROCKET_MASS, ROCKET_THRUST, ROCKET_TORQUE)
dqn_controller = DQNController(dqn_rocket, (650, TARGET_Y),
                               filename='models/hover-complex-assist-fixtarget-fixrocket-ep2000.h5', control_type='assisted')

game.run()
