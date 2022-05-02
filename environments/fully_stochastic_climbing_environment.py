import random
from environments.environment_interface import EnvironmentInterface


class FullyStochasticClimbingEnvironment(EnvironmentInterface):

    def __init__(self):
        a = "a"
        b = "b"
        c = "c"

        self.actions = [a, b, c]
        self.num_states = 10
        self.terminal_states = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.ylim = [-10, 12]

        self.t_func = {
            (0, a, a): [[10, 12], 1],
            (0, a, b): [[5, -65], 2],
            (0, a, c): [[8, -8], 3],

            (0, b, a): [[5, -65], 4],
            (0, b, b): [[14, 0], 5],
            (0, b, c): [[12, 0], 6],

            (0, c, a): [[8, -8], 7],
            (0, c, b): [[12, 0], 8],
            (0, c, c): [[10, 0], 9],
        }

    # returns new a1_reward, a1_state, a2_reward, a2_state
    def reward_function(self, a1_state: int, a1_action: int, a2_state: int, a2_action: int) -> (int, int, int, int):
        result = self.t_func[(a1_state, self.actions[a1_action], self.actions[a2_action])]
        reward = result[0]
        if type(reward) is list:
            rand_reward = random.choice(reward)
            return rand_reward, result[1], rand_reward, result[1]
        else:
            return reward, result[1], reward, result[1]
