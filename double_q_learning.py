import random

class DoubleQLearning:

    def __init__(self, alpha, epsilon, tau, gamma, num_actions, num_states, start_state=0):
        self.alpha = alpha
        self.epsilon = epsilon
        self.tau = tau
        self.gamma = gamma
        self.num_actions = num_actions
        self.num_states = num_states
        self.start_state = start_state

        self.cur_state = start_state
        self.cur_action = None

        self.q_table_a = [[0] * self.num_actions for _ in range(0, self.num_states)]
        self.q_table_b = [[0] * self.num_actions for _ in range(0, self.num_states)]
        self.merged_q_table = []

    @staticmethod
    def get_name():
        """Return name of the action selection method for plotting"""
        pass

    # Cause agent to select an action and return that action
    def get_action(self) -> int:
        """Select action, abstract for extendability by either softmax, egreedy, or random"""
        pass

    def get_state(self) -> int:
        return self.cur_state

    def reset(self):
        self.cur_state = self.start_state

    # Environment gives agent a new state and reward for taking an action
    def reward(self, reward, new_state):
        rand_int = random.randint(0, 1)
        # Update Q-table A
        if rand_int == 1:
            a_max = max(self.q_table_a[self.cur_state])
            a_act = self.q_table_a[self.cur_state].index(a_max)
            self.q_table_a[self.cur_state][self.cur_action] = self.q_table_a[self.cur_state][self.cur_action] + \
                                                              self.alpha * \
                                                              (reward + (self.gamma * self.q_table_b[new_state][a_act]) -
                                                                      self.q_table_a[self.cur_state][self.cur_action])
        else:
            b_max = max(self.q_table_b[self.cur_state])
            b_act = self.q_table_b[self.cur_state].index(b_max)
            self.q_table_b[self.cur_state][self.cur_action] = self.q_table_b[self.cur_state][self.cur_action] + self.alpha * (
                    reward + (self.gamma * self.q_table_a[new_state][b_act]) - self.q_table_b[self.cur_state][
                self.cur_action])

        self.cur_state = new_state