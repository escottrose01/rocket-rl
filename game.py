import pygame
import numpy as np

from engine import GameObject, Game, Physics
from controller import RocketController, PlayerController, PIDController, OnOffController, DQNController
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

target = (650, TARGET_Y)

myrocket = Rocket((600, GROUND_Y), ROCKET_MASS, ROCKET_THRUST, ROCKET_TORQUE)
plane = PlaneGravitySource(GRAVITY, GROUND_Y, GROUND_BOUNCE, GROUND_FRICTION)
Circle(*target, 5)

mycontroller = PlayerController(myrocket)
game.register_callback(lambda _, __: mycontroller.accept_input(pygame.key.get_pressed()))

for i in range(1, 10):
    rocket = Rocket((100 + i*50, GROUND_Y), ROCKET_MASS, ROCKET_THRUST, ROCKET_TORQUE)
    kp = np.random.uniform(0.0, 1.0)
    ki = np.random.uniform(0.0, 0.001)
    kd = np.random.uniform(0.0, 1.0)
    controller = PIDController(rocket, target, kp, ki, kd)

dqn_rocket = Rocket((650, GROUND_Y-100), ROCKET_MASS, ROCKET_THRUST, ROCKET_TORQUE)
dqn_controller = DQNController(dqn_rocket, (600, TARGET_Y), filename='models/hover-simple-final.h5')

game.run()
