import pygame

from gym import Env
from gym.spaces import Box, Discrete
import numpy as np

import h5py

from engine import Game
from rocket import Rocket
from controller import PlayerController
from planet import PlaneGravitySource
from util import Circle, Text
from policy import RepEpsGreedyPolicy

GROUND_Y = 750
GROUND_BOUNCE = 0.4
GROUND_FRICTION = 2.5
TARGET_Y = 200

GRAVITY = 9.8

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class LearningEnv(Env):
    """A learning environment for a rocket controller."""

    def __init__(self):
        Game.instance().graphics = False
        self.timestep = 1/60
        self.rounds = 10000
        self.collected_reward = 0
        self.state = None

        self._rocket, self._controller = self.build_rocket()

    def step(self, action):
        done = False
        info = {}
        rw = 0
        self.rounds -= 1

        assert self.action_space.contains(action), "Invalid Action"

        self.apply_action(action)
        Game.instance().step(self.timestep)

        rw = self.get_reward()
        obs = self.get_obs()
        done = self.is_done()

        self.collected_reward += rw

        return obs, rw, done, info

    def reset(self):
        Game.instance().reset()

        self._rocket, self._controller = self.build_rocket()
        self._target = self.make_target()
        self.rounds = 10000
        self.collected_reward = 0
        self.state = None

    def get_obs(self):
        raise NotImplementedError()

    def get_reward(self):
        raise NotImplementedError()

    def is_done(self):
        raise NotImplementedError()

    def apply_action(self, action):
        raise NotImplementedError()

    def build_rocket(self) -> tuple[Rocket, PlayerController]:
        raise NotImplementedError()

    def make_target(self):
        raise NotImplementedError()

    def render(self, *args, **kwargs):
        Game.instance().render()


class SimpleLearningEnv(LearningEnv):
    def __init__(self, resolution=10, fix_target=False, fix_rocket=False):
        super().__init__()

        self._w = 800
        self._h = 800
        self._box = np.array([self._w, self._h])

        self._fix_rocket = fix_rocket
        self._fix_target = fix_target

        self.reset()

        # initialize learning environment
        self._res = resolution
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=self.get_obs().shape)
        self.action_space = Discrete(resolution+1)

    def step(self, action):
        obs, rw, done, info = super().step(action)

        # update text
        self._reward_text.set_text(f'reward: {self.collected_reward:.3f}')
        self._position_text.set_text(f'position: ({self._rocket.position[0]:.0f}, {self._rocket.position[1]:.0f})')
        self._velocity_text.set_text(f'velocity: ({self._rocket.velocity[0]:.1f}, {self._rocket.velocity[1]:.1f})')
        self.heading_text.set_text(f'ang. vel.: {self._rocket.angular_velocity:.1f}, torque: {self._rocket.torque:.1f}')
        self._thrust_text.set_text(f'thrust: {self._rocket.thrust:.2f}')

        return obs, rw, done, info

    def reset(self):
        super().reset()

        self._plane = PlaneGravitySource(GRAVITY, GROUND_Y, GROUND_BOUNCE, GROUND_FRICTION)

        self._old_err = self._rocket.position - self._target
        self._cum_err = 0.0
        self._old_err_obs = self._rocket.position_obs - self._target
        self._cum_err_obs = 0.0

        Circle(*self._target, 5)
        self._reward_text = Text(10, 10, 'reward: -----')
        self._position_text = Text(10, 36, 'position: (----- -----)')
        self._velocity_text = Text(10, 62, 'velocity: (-----, -----)')
        self.heading_text = Text(10, 88, 'ang. vel.: -----, torque: -----')
        self._thrust_text = Text(10, 114, 'thrust: -----')

        self.rounds = 10000
        self.collected_reward = 0

        return self.get_obs()

    def get_reward(self):
        # reward proximity to target
        err = (self._target[1] - self._rocket.position[1]) / self._h
        rw = -0.001 * self.timestep
        if abs(err) < 0.2:
            rw += 2.0 * self.timestep
        if abs(err) < 0.1:
            rw += 1.0 * self.timestep
        if self._rocket.grounded:
            rw -= 0.1 * self.timestep

        return rw

    def is_done(self):
        # return self._rocket.crashed or self.rounds == 0 or self._rocket.position[1] < -2*self._h
        return self.rounds == 0 or self._rocket.crashed

    def apply_action(self, action):
        self._controller.thrust = action / self._res

    def build_rocket(self):
        mass = 27670.
        max_thrust = 410000.
        max_torque = 25.
        # pos = np.array([400, GROUND_Y - np.random.uniform(0, 200)])
        if self._fix_rocket:
            pos = np.array([400, 400])
            vel = np.array([0, 0])
        else:
            pos = np.array([400, np.random.uniform(0, 400.0 + np.random.uniform(-100.0, 100.0))])
            vel = np.array([0, np.random.uniform(-20.0, 20.0)])

        rocket = Rocket(pos, mass, max_thrust, max_torque)
        rocket.velocity = vel
        controller = PlayerController(rocket)

        return rocket, controller

    def make_target(self):
        if self._fix_target:
            return np.array([400, 400])
        else:
            return np.array([400, np.random.uniform(0, 400.0 + np.random.uniform(-100.0, 100.0))])

    def get_obs(self):
        position = self._rocket.position_obs
        velocity = self._rocket.velocity_obs
        heading = self._rocket.heading_obs
        height = position[1] - GROUND_Y
        target = self._target
        err_k = (target - position)
        err_d = (err_k - self._old_err_obs) / self.timestep
        err_i = self._cum_err_obs + err_k * self.timestep
        return np.array([height, *velocity, *heading, *target,
                        *err_k, *err_d, *err_i,
                         # velocity[0], velocity[1],
                         #  heading[0], heading[1],
                         #  target[0], target[1],
                         #  err_k[0], err_k[1],
                         #  err_d[0], err_d[1],
                         #  err_i[0], err_i[1],
                         self._rocket.rotation_obs,
                         self._rocket.angular_velocity])

    def render(self, *args, **kwargs):
        if self.rounds % 20 == 0:
            Game.instance().render()


class ComplexLearningEnv(SimpleLearningEnv):
    def __init__(self):
        super().__init__()

        self._w = 800
        self._h = 600
        self._box = np.array([self._w, self._h])

        # initialize learning environment
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=self.get_obs().shape)
        self.action_space = Discrete(2*3)

        self.reset()

    def make_target(self):
        return np.array([np.random.uniform(0, self._w), np.random.uniform(0, GROUND_Y-100)])

    def get_reward(self):
        # reward proximity to target
        err = 1/self._box * (self._target - self._rocket.position)
        d = np.linalg.norm(err)
        rw += 0.2 * (1.0 - d)

        # reward movement towards target from far away
        v = self._rocket.velocity / self._box
        rw += 0.1 * d * np.inner(v, err)

        # reward orientation and motion of rocket
        if abs(self._rocket.rotation) < np.pi/6:
            rw += 0.1
        else:
            rw += 0.1 * (1.0 - abs(self._rocket.rotation) / (0.5 * np.pi))

        if abs(self._rocket.rotation) > np.pi/2:
            rw = min(rw, 0)
        if abs(self._rocket.angular_velocity) > np.pi:
            rw = min(rw, 0)

        # scale reward
        return rw * self.timestep

    def apply_action(self, action):
        torque = action // 2 - 1
        thrust = action % 2

        self._controller.torque = min(1.0, max(-1.0, torque))
        self._controller.thrust = min(1.0, max(0.0, thrust))


if __name__ == '__main__':
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Flatten
    from keras.optimizers import Adam
    from keras.callbacks import LambdaCallback

    from rl.agents.dqn import DQNAgent
    from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
    from rl.memory import SequentialMemory

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    env = SimpleLearningEnv()

    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), 'eps', 0.1, 0.01, 0.01, 100000)
    memory = SequentialMemory(limit=500000, window_length=1)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
                   target_model_update=1e-2, policy=policy, gamma=0.99,
                   enable_double_dqn=True, enable_dueling_network=True)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    checkpoint = LambdaCallback(on_epoch_end=lambda epoch, logs: epoch %
                                200 == 0 and dqn.model.save(f'models/hover-example-ep{epoch}.h5'))

    dqn.fit(env, nb_steps=50000, visualize=True, verbose=1,
            log_interval=10000, callbacks=[checkpoint], action_repetition=10)
    dqn.model.save('models/hover-example-final.h5')
