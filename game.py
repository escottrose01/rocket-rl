import pygame
import numpy as np

from engine import Game, HEIGHT
from controller import PlayerController, PIDController, DQNController
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
camera = KeyboardCamera()

all_sprites = pygame.sprite.Group()

target = (650, TARGET_Y)

myrocket = Rocket((600, GROUND_Y), ROCKET_MASS, ROCKET_THRUST, ROCKET_TORQUE)
plane = PlaneGravitySource(GRAVITY, GROUND_Y, GROUND_BOUNCE, GROUND_FRICTION)
Circle(*target, 5)

mycontroller = PlayerController(myrocket)
game.register_callback(lambda _, __: mycontroller.accept_input(pygame.key.get_pressed()))
game.register_callback(lambda *_: camera.accept_input(pygame.key.get_pressed()))

for i in range(1, 10):
    rocket = Rocket((100 + i*50, GROUND_Y), ROCKET_MASS, ROCKET_THRUST, ROCKET_TORQUE)
    kp = 1.0
    ki = 0.001
    kd = 0.3 * i
    controller = PIDController(rocket, target, kp, ki, kd)

dqn_rocket = Rocket((650, GROUND_Y), ROCKET_MASS, ROCKET_THRUST, ROCKET_TORQUE)
dqn_controller = DQNController(dqn_rocket, (650, TARGET_Y), filename='models/hover-example-final.h5')

game.run()
