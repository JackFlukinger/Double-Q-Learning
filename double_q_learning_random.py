import random
from double_q_learning import DoubleQLearning


class DoubleQLearningRandom(DoubleQLearning):

    @staticmethod
    def get_name():
        return "Random"

    # Select actions randomly
    def get_action(self):
        # Return random action
        self.cur_action = random.randint(0, self.num_actions - 1)
        return self.cur_action
