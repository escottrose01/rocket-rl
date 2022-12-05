from gym import Env
from gym.spaces import Box, Discrete
import numpy as np

from engine import Game, Camera
from rocket import Rocket
from controller import PlayerController
from planet import PlaneGravitySource
from util import Circle, Text
from camera import FollowCamera

GROUND_Y = 0
GROUND_BOUNCE = 0.4
GROUND_FRICTION = 2.5

GRAVITY = 9.8


class LearningEnv(Env):
    """A learning environment for a rocket controller."""

    def __init__(self):
        Game.instance().graphics = False
        self.timestep = 1/60
        self.rounds = 10000
        self.collected_reward = 0
        self.reward_history = []
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

        if done:
            self.reward_history.append(self.collected_reward)

        return obs, rw, done, info

    def reset(self):
        Game.instance().reset()

        self._rocket, self._controller = self.build_rocket()
        self._target = self.make_target()
        self.rounds = 10000
        self.collected_reward = 0
        self.state = None
        Camera(y=300-25)

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


class SimpleHoverEnv(LearningEnv):
    def __init__(self, resolution=10, fix_target=False, fix_rocket=False):
        self._fix_rocket = fix_rocket
        self._fix_target = fix_target

        super().__init__()

        self._w = 800
        self._h = 800
        self._box = np.array([self._w, self._h])

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
        err = (self._target[1] - self._rocket.position[1])
        rw = -0.001 * self.timestep
        if abs(err) < 150:
            rw += 1.0 * self.timestep
        if abs(err) < 50:
            rw += 1.0 * self.timestep
        if self._rocket.grounded:
            rw -= 0.1 * self.timestep

        return rw

    def is_done(self):
        return self.rounds == 0 or self._rocket.crashed

    def apply_action(self, action):
        self._controller.thrust = action / self._res

    def build_rocket(self):
        mass = 27670.
        max_thrust = 410000.
        max_torque = 25.
        if self._fix_rocket:
            pos = np.array([400, 400])
            vel = np.array([0, 0])
        else:
            pos = np.array([400, 400.0 + np.random.uniform(-200.0, 300.0)])
            vel = np.array([0, np.random.uniform(-30.0, 30.0)])

        rocket = Rocket(pos, mass, max_thrust, max_torque)
        rocket.velocity = vel
        controller = PlayerController(rocket)

        return rocket, controller

    def make_target(self):
        if self._fix_target:
            return np.array([400, 400])
        else:
            return np.array([400, 400.0 + np.random.uniform(-200.0, 300.0)])

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
                         self._rocket.rotation_obs,
                         self._rocket.angular_velocity])

    def render(self, *args, **kwargs):
        if self.rounds % 20 == 0:
            Game.instance().render()


class ComplexHoverEnv(SimpleHoverEnv):
    def __init__(self, fix_target=False, fix_rocket=False):
        self._w = 800
        self._h = 600
        self._box = np.array([self._w, self._h])

        super().__init__(fix_target=fix_target, fix_rocket=fix_rocket)

        # initialize learning environment
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=self.get_obs().shape)
        self.action_space = Discrete(11*5)

        self.reset()

    def get_reward(self):
        # reward survival
        rw = 0.01 * self.timestep

        # reward proximity to target
        err = self._target - self._rocket.position
        d = np.linalg.norm(err)
        if d < 300:
            rw += 0.1 * self.timestep
        if d < 150:
            rw += 1.0 * self.timestep
        if d < 50:
            rw += 1.0 * self.timestep

        # reward positive orientation
        if self._rocket.heading[1] > 0.85 and self._rocket.velocity[1] > -50.0:
            rw += 0.1 * self.timestep

        # punish unstable behaviors
        if abs(self._rocket.angular_velocity) > 1.5:
            rw = -1.0 * self.timestep
        # elif self._rocket.heading[1] < 0:
        #     rw = 0

        return rw

    def apply_action(self, action):
        torque = (action // 11 - 2) / 2.0
        thrust = 0.1*(action % 11)

        self._controller.torque = min(1.0, max(-1.0, torque))
        self._controller.thrust = min(1.0, max(0.0, thrust))

    def build_rocket(self):
        mass = 27670.
        max_thrust = 410000.
        max_torque = 25.
        if self._fix_rocket:
            pos = np.array([400.0, 500.0])
            # vel = np.array([0.0, 0.0])
            vel = np.array([np.random.uniform(-1.0, 1.0), 0])
        else:
            pos = np.array([np.random.uniform(0, self._w), np.random.uniform(200.0, 700.0)])
            vel = np.array([np.random.uniform(-5.0, 5), np.random.uniform(-30.0, 30.0)])

        rocket = Rocket(pos, mass, max_thrust, max_torque)
        rocket.velocity = vel
        controller = PlayerController(rocket)

        return rocket, controller

    def make_target(self):
        if self._fix_target:
            return np.array([400, 500])
        else:
            return np.array([np.random.uniform(0, self._w), np.random.uniform(200.0, 700.0)])


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

    env = SimpleHoverEnv()

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

    dqn.fit(env, nb_steps=500000, visualize=True, verbose=1,
            log_interval=10000, callbacks=[checkpoint], action_repetition=10)
    dqn.model.save('models/hover-example-final.h5')
