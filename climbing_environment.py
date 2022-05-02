from environment_interface import EnvironmentInterface


class ClimbingEnvironment(EnvironmentInterface):
    a = "a"
    b = "b"
    c = "c"

    actions = [a, b, c]
    num_states = 2
    terminal_states = [1]
    ylim = [-10,12]

    t_func = {
        (a, a): 11,
        (a, b): -30,
        (a, c): 0,

        (b, a): -30,
        (b, b): 7,
        (b, c): 6,

        (c, a): 0,
        (c, b): 0,
        (c, c): 5,
    }

    # returns new a1_reward, a1_state, a2_reward, a2_state
    def reward_function(self, a1_state: int, a1_action: int, a2_state: int, a2_action: int) -> (int, int, int, int):
        reward = self.t_func[(self.actions[a1_action], self.actions[a2_action])]
        return reward, 1, reward, 1
