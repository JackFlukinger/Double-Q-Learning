import random
from q_learning.double_q_learning import DoubleQLearning


class DoubleQLearningEGreedy(DoubleQLearning):

    @staticmethod
    def get_name():
        return "E-Greedy"

    # Select actions using epsilon greedy while handling the exploration-exploitation dilemma
    def get_action(self):
        self.time += 1

        merged_q_table = [(x + y) / 2 for x, y in zip(self.q_table_a[self.cur_state], self.q_table_b[self.cur_state])]

        real_epsilon = self.epsilon
        if self.epsilon == "1/T":
            real_epsilon = 1 / self.time

        # Generate random number between 0 and 1
        rand = random.random()
        if rand < real_epsilon:
            self.cur_action = random.randint(0, self.num_actions - 1)
            return self.cur_action
        max_value = max(merged_q_table)
        self.cur_action = merged_q_table.index(max_value)
        return self.cur_action

    def optimal_action(self, state):

        merged_q_table = [(x + y) / 2 for x, y in zip(self.q_table_a[state], self.q_table_b[state])]
        optimal_action = merged_q_table.index(max(merged_q_table))

        return optimal_action