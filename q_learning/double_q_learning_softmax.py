import random
from q_learning.double_q_learning import DoubleQLearning


class DoubleQLearningSoftmax(DoubleQLearning):

    @staticmethod
    def get_name():
        return "Softmax"

    def get_action(self):
        self.time += 1

        merged_q_table = [(x + y) / 2 for x, y in zip(self.q_table_a[self.cur_state], self.q_table_b[self.cur_state])]
        e = 2.71828
        population = list(range(0, self.num_actions))
        weights = list(range(0, self.num_actions))
        denom = 0
        for action in range(0, self.num_actions):
            denom += e ** (merged_q_table[action] / (self.tau * (self.delta ** self.time)))
        for action in range(0, self.num_actions):
            num = e ** (merged_q_table[action] / (self.tau * (self.delta ** self.time)))
            weights[action] = num / denom
        self.cur_action = random.choices(population, weights)[0]
        return self.cur_action

    def optimal_action(self, state):

        merged_q_table = [(x + y) / 2 for x, y in zip(self.q_table_a[state], self.q_table_b[state])]
        optimal_action = merged_q_table.index(max(merged_q_table))

        return optimal_action
