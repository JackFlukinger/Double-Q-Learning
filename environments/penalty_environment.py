from environments.environment_interface import EnvironmentInterface


class PenaltyEnvironment(EnvironmentInterface):

    def __init__(self, k):
        a = "a"
        b = "b"
        c = "c"
        self.t_func = {
            (a, a): 10,
            (a, b): 0,
            (a, c): k,

            (b, a): 0,
            (b, b): 2,
            (b, c): 0,

            (c, a): k,
            (c, b): 0,
            (c, c): 10,
        }
        self.actions = [a, b, c]
        self.num_states = 2
        self.terminal_states = [1]
        self.ylim = [0, 12]

    # returns new a1_reward, a1_state, a2_reward, a2_state
    def reward_function(self, a1_state: int, a1_action: int, a2_state: int, a2_action: int) -> (int, int, int, int):
        reward = self.t_func[(self.actions[a1_action], self.actions[a2_action])]
        return reward, 1, reward, 1