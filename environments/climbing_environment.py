from environments.environment_interface import EnvironmentInterface


class ClimbingEnvironment(EnvironmentInterface):

    def __init__(self):
        a = "a"
        b = "b"
        c = "c"

        self.actions = [a, b, c]
        self.num_states = 10
        self.terminal_states = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.ylim = [-10,12]

        self.t_func = {
            (0, a, a): [11, 1],
            (0, a, b): [-30, 2],
            (0, a, c): [0, 3],

            (0, b, a): [-30, 4],
            (0, b, b): [7, 5],
            (0, b, c): [6, 6],

            (0, c, a): [0, 7],
            (0, c, b): [0, 8],
            (0, c, c): [5, 9],
        }

    # returns new a1_reward, a1_state, a2_reward, a2_state
    def reward_function(self, a1_state: int, a1_action: int, a2_state: int, a2_action: int) -> (int, int, int, int):
        result = self.t_func[(a1_state, self.actions[a1_action], self.actions[a2_action])]
        return result[0], result[1], result[0], result[1]
