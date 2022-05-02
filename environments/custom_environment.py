from environments.environment_interface import EnvironmentInterface


class CustomEnvironment(EnvironmentInterface):

    def __init__(self):
        self.stay = "stay"
        self.forward = "forward"

        self.actions = [self.forward, self.stay]
        self.num_states = 12
        self.terminal_states = [4, 11]
        self.ylim = [-10, 12]

    def t_func(self, a1_state, a1_action, a2_state, a2_action):
        new_a1_state = a1_state
        new_a2_state = a2_state
        if a1_action == self.forward and a2_action == self.forward:
            new_a1_state += 2
            new_a2_state += 2
            if new_a1_state > 11:
                new_a1_state = 11
            if new_a2_state > 11:
                new_a2_state = 11
        if new_a1_state == 11 or new_a2_state == 11:
            return 100, new_a1_state, 100, new_a2_state
        elif new_a1_state == 5 or new_a2_state == 5:
            return -10, new_a1_state, -10, new_a2_state
        elif a1_action == self.stay and a2_action == self.stay:
            return -10, 4, -10, 4
        else:
            a1_reward = 1
            a2_reward = 1
            if a1_action == self.forward:
                new_a1_state += 1
            if a2_action == self.forward:
                new_a2_state += 1
            if a1_action == self.stay:
                a1_reward = 0
            if a2_action == self.stay:
                a2_reward = 0
            return a1_reward, new_a1_state, a2_reward, new_a2_state


    # returns new a1_reward, a1_state, a2_reward, a2_state
    def reward_function(self, a1_state: int, a1_action: int, a2_state: int, a2_action: int) -> (int, int, int, int):
        result = self.t_func(a1_state, self.actions[a1_action], a2_state, self.actions[a2_action])
        #print(result)
        return result[0], result[1], result[2], result[3]
