import random
from q_learning.double_q_learning import DoubleQLearning


class DoubleQLearningRandom(DoubleQLearning):

    @staticmethod
    def get_name():
        return "Random"

    # Select actions randomly
    def get_action(self):
        self.time += 1

        # Return random action
        self.cur_action = random.randint(0, self.num_actions - 1)
        return self.cur_action

    # Optimal action is random for random agent
    def optimal_action(self, state):

        optimal_action = random.randint(0, self.num_actions - 1)

        return optimal_action