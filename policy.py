import numpy as np
from rl.policy import EpsGreedyQPolicy


class RepEpsGreedyPolicy(EpsGreedyQPolicy):
    def __init__(self, eps=0.1, rep=20):
        super().__init__(eps)
        self._steps = 0
        self._rep = rep

    def select_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if self._steps > 0:
            self._steps -= 1
            action = np.random.randint(0, nb_actions)
        elif np.random.uniform() < self.eps:
            self._steps = self._rep
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(q_values)
        return action
