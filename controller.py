"""A controller for a rocket."""

import numpy as np
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    K_x,
    K_z,
    K_f,
    K_LSHIFT,
    K_RSHIFT,
    K_LCTRL,
    K_RCTRL,
    KEYDOWN,
    QUIT,
)

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

from engine import UpdateListener
from rocket import Rocket, TrajectoryInfo
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def sigmoid(x: float) -> float:
    """Returns the sigmoid function applied to x.
    This function is computed in a numerically stable way.

    Args:
        x (float): the input.

    Returns:
        float: sigmoid(x).
    """
    return np.exp(-np.logaddexp(0, -x))


class RocketController(UpdateListener):
    """An absract base class for a rocket controller."""

    def __init__(self, rocket: Rocket):
        """Initializes a new RocketController instance.

        Args:
            rocket (Rocket): the rocket object to control.
        """
        super().__init__()

        self._rocket = rocket
        self._cur_info = rocket.trajectory_info
        self._prev_info = None

    def on_fixed_update(self, dt):
        self._prev_info = self._cur_info
        self._cur_info = self._rocket.trajectory_info


class PlayerController(RocketController):
    """A class for controlling a rocket via player input"""

    def __init__(self, rocket: Rocket):
        """Initializes a new PlayerController instance.

        Args:
            rocket (Rocket): the rocket object to control.
        """
        super().__init__(rocket)

        self._thrust = 0.
        self._torque = 0.

    @property
    def thrust(self) -> float:
        """Returns the current thrust of this controller.

        Returns:
            float: the thrust of this controller, in the range [0, 1].
        """
        return self._thrust

    @thrust.setter
    def thrust(self, value):
        """Sets the current thrust of this controller.

        Args:
            value (float): the thrust of this controller, as a percentage of its maximum thrust.
        """
        self._thrust = min(max(value, 0.0), 1.0)

    @property
    def torque(self) -> float:
        """Returns the current torque of this controller.

        Returns:
            float: the thrust of this controller, as a signed percentage of its maximum torque.
        """
        return self._torque

    @torque.setter
    def torque(self, value):
        """Sets the current torque of this controller.

        Args:
            value (float): the torque of this controller, as a signed percentage of its maximum torque.
        """
        self._torque = min(max(value, -1.0), 1.0)

    def on_fixed_update(self, dt: float):
        self._rocket.thrust = self._thrust
        self._rocket.torque = self._torque

    def accept_input(self, pressed_keys: list):
        """Accepts input from the player.

        Args:
            pressed_keys (list): the list of keys currently pressed.
        """
        if pressed_keys[K_UP]:
            pass
        if pressed_keys[K_DOWN]:
            pass
        if pressed_keys[K_LEFT]:
            self._torque = -1.0
        elif pressed_keys[K_RIGHT]:
            self._torque = 1.0
        else:
            if abs(self._rocket.angular_velocity) > 1e-5:
                self._torque = 1.0 if self._rocket.angular_velocity < 0 else -1.0
            else:
                self._torque = 0.0
        if pressed_keys[K_LCTRL] or pressed_keys[K_RCTRL]:
            self.thrust -= 0.01
        if pressed_keys[K_LSHIFT] or pressed_keys[K_RSHIFT]:
            self.thrust += 0.01
        if pressed_keys[K_x]:
            self._thrust = 0.0
        if pressed_keys[K_z]:
            self._thrust = 1.0


class HoverRocketController(RocketController):
    """A base class for a rocket controller operating in the hover environment."""

    def __init__(self, rocket: Rocket, target: list = None):
        """Initializes a new HoverRocketController instance.

        Args:
            rocket (Rocket): the rocket object to control.
            target (list, optional): the target position to reach. If None,
                sets the target as the current rocket position. Defaults to None.
        """
        super().__init__(rocket)
        self._target = np.array(target) if target else rocket.position

    @property
    def error(self) -> np.ndarray:
        """Returns the vector error between the target and current rocket positions.

        Returns:
            np.ndarray: the vector error between target and current positions.
        """
        return self._target - self._rocket.position_obs


class OnOffController(HoverRocketController):
    """An on-off hover controller.

    This simple controller simply returns a binary on/off signal
    depending on whether the rocket is above or below the target
    elevation. This controller ignores the target horizontal
    displacement.
    """

    def on_fixed_update(self, dt: float):
        super().on_fixed_update(dt)

        if self.error[1] > 0:
            self._rocket.thrust = 0.
        else:
            self._rocket.thrust = 1.


class PIDController(HoverRocketController):
    """A proportional-integral-derivative (PID) hover controller.

    Uses a weighted combination of the error, error integral, and error
    derivative to control the rocket.
    """

    def __init__(self, rocket: Rocket, target: list = None, kp: float = 1., ki: float = 0., kd: float = 1.):
        """Initializes a new PIDController Instance.

        Args:
            rocket (Rocket): the rocket object to control.
            target (list, optional): the target position to reach If None,
                sets the target as the current rocket position. Defaults to None.
            kp (float, optional): the proportional weight constant. Defaults to 1.
            ki (float, optional): the integral weight constant. Defaults to 0.
            kd (float, optional): the derivative weight constant. Defaults to 1.
        """
        super().__init__(rocket, target)

        self._kp = kp
        self._ki = ki
        self._kd = kd

        self._err_y_prev = 0.
        self._err_y_cum = 0.
        self.err_x_prev = 0.
        self.err_x_cum = 0.

    def on_fixed_update(self, dt: float):
        super().on_fixed_update(dt)

        error_y = -self.error[1]
        self._err_y_cum += error_y * dt
        derivative_y = (error_y - self._err_y_prev) / dt
        self._err_y_prev = error_y

        self._rocket.thrust = 1.0-sigmoid(self._kp * error_y + self._ki * self._err_y_cum + self._kd * derivative_y)


class DQNController(HoverRocketController):
    """A DQN hover controller"""

    # TODO(escottrose01): rewrite observation to not require box size
    GROUND_Y = 550

    def __init__(self, rocket: Rocket, target: list, filename: str, control_type='simple', query_interval=20):
        super().__init__(rocket, target)

        self._model = keras.models.load_model(filename)
        self._rocket = rocket
        self._control_type = control_type
        self.query_interval = query_interval
        self._iter = 0
        self._action = None
        self._nb_actions = self._model.layers[-1].output_shape[1]

        self._old_err_obs = self._target - self._rocket.position_obs
        self._cum_err_obs = 0.0

    def on_fixed_update(self, dt):
        super().on_fixed_update(dt)

        if self._iter == 0:
            self._iter = self.query_interval
            obs = self._get_obs(dt).reshape((1, 1, -1))
            pred = self._model.predict(obs, verbose=False).reshape((-1,)).tolist()
            self._action = np.argmax(pred)
        else:
            self._iter -= 1

        if self._control_type == 'simple':
            thrust = self._action / self._nb_actions
            self._rocket.thrust = thrust
        elif self._control_type == 'complex':
            torque = self._action // 2 - 1
            thrust = self._action % 2

            self._rocket.torque = min(1.0, max(-1.0, torque))
            self._rocket.thrust = min(1.0, max(0.0, thrust))
        else:
            raise ValueError()

        self._old_err_obs = self._rocket.position_obs - self._target
        self._cum_err_obs += dt * (self._target - self._rocket.position_obs)

    def _get_obs(self, dt):
        position = self._rocket.position_obs
        velocity = self._rocket.velocity_obs
        heading = self._rocket.heading_obs
        height = position[1] - DQNController.GROUND_Y
        target = self._target
        err_k = (target - position)
        err_d = (err_k - self._old_err_obs) / dt
        err_i = self._cum_err_obs + err_k * dt
        return np.array([height, *velocity, *heading, *target,
                        *err_k, *err_d, *err_i,
                         self._rocket.rotation_obs,
                         self._rocket.angular_velocity])
