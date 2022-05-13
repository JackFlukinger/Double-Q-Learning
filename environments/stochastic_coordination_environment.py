import random
from environments.environment_interface import EnvironmentInterface


class StochasticCoordinationEnvironment(EnvironmentInterface):

    def __init__(self, k):
        a = "a"
        b = "b"

        self.name = "Stochastic Coordination Game k=" + str(k)

        self.actions = [a, b]
        self.num_states = 6
        self.terminal_states = [3, 4, 5]
        self.ylim = [-10, 12]

        self.t_func = {
            # (state, p1 action, p2 action): [reward, next_state]
            (0, a, a): [0, 1],
            (0, a, b): [0, 1],

            (0, b, a): [0, 2],
            (0, b, b): [0, 2],

            (1, a, a): [[5, 17], 3],
            (1, b, b): [[5, 17], 3],

            (1, a, b): [[0, k], 4],
            (1, b, a): [[0, k], 4],

            (2, a, a): [[4, 10], 5],
            (2, a, b): [[4, 10], 5],
            (2, b, a): [[4, 10], 5],
            (2, b, b): [[4, 10], 5],
        }

        self.optimal = 11

        if k > 11:
            self.optimal = k
            self.ylim = [0, k + 10]

    # returns new a1_reward, a1_state, a2_reward, a2_state
    def reward_function(self, a1_state: int, a1_action: int, a2_state: int, a2_action: int) -> (int, int, int, int):
        result = self.t_func[(a1_state, self.actions[a1_action], self.actions[a2_action])]
        reward = result[0]
        if type(reward) is list:
            rand_reward = random.choice(reward)
            return rand_reward, result[1], rand_reward, result[1]
        else:
            return reward, result[1], reward, result[1]
