import random
from double_q_learning import DoubleQLearning


class DoubleQLearningEGreedy(DoubleQLearning):

    @staticmethod
    def get_name():
        return "E-Greedy"

    # Select actions using epsilon greedy while handling the exploration-exploitation dilemma
    def get_action(self):
        self.merged_q_table = [[(x + y) / 2 for x, y in zip(self.q_table_a[self.cur_state], self.q_table_b[self.cur_state])]]
        # Generate random number between 0 and 1
        rand = random.random()
        if rand < self.epsilon:
            self.cur_action = random.randint(0, self.num_actions - 1)
            return self.cur_action
        #print(self.merged_q_table)
        max_value = max(self.merged_q_table[self.cur_state])
        self.cur_action = self.merged_q_table[self.cur_state].index(max_value)
        return self.cur_action
