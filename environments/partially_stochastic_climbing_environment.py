import random
from environments.environment_interface import EnvironmentInterface

class PartiallyStochasticClimbingEnvironment(EnvironmentInterface):

    def __init__(self):
        a = "a"
        b = "b"
        c = "c"

        self.actions = [a, b, c]
        self.num_states = 2
        self.terminal_states = [1]
        self.ylim = [-10,12]

        self.t_func = {
            (a, a): 11,
            (a, b): -30,
            (a, c): 0,

            (b, a): -30,
            (b, b): [14, 0],
            (b, c): 6,

            (c, a): 0,
            (c, b): 0,
            (c, c): 5,
        }

    # returns new a1_reward, a1_state, a2_reward, a2_state
    def reward_function(self, a1_state: int, a1_action: int, a2_state: int, a2_action: int) -> (int, int, int, int):
        reward = self.t_func[(self.actions[a1_action], self.actions[a2_action])]
        if type(reward) is list:
            rand_reward = random.choice(reward)
            return rand_reward, 1, rand_reward, 1
        else:
            return reward, 1, reward, 1
