import matplotlib.pyplot as plt

from double_q_learning import DoubleQLearning
from double_q_learning_egreedy import DoubleQLearningEGreedy
from double_q_learning_random import DoubleQLearningRandom
from double_q_learning_softmax import DoubleQLearningSoftmax


class EnvironmentInterface:
    actions = []
    terminal_states = []
    ylim = [-50, 50]
    num_states = 0
    figure, axis = plt.subplots(2, 2)

    def reward_function(self, p1_state: int, p1_action: int, p2_state: int, p2_action: int) -> (int, int, int, int):
        """Return tuple (p1_reward, p1_action, p2_reward, p2_action) for agents taking actions"""
        pass

    def run_tests(self, alpha, epsilon, tau, gamma, episodes, simulations=1):
        self.run_test((0,0), DoubleQLearningSoftmax, DoubleQLearningSoftmax, alpha, epsilon, tau, gamma, episodes, simulations)
        self.run_test((0,1), DoubleQLearningSoftmax, DoubleQLearningRandom, alpha, epsilon, tau, gamma, episodes, simulations)
        self.run_test((1,0), DoubleQLearningEGreedy, DoubleQLearningEGreedy, alpha, epsilon, tau, gamma, episodes, simulations)
        self.run_test((1,1), DoubleQLearningEGreedy, DoubleQLearningRandom, alpha, epsilon, tau, gamma, episodes, simulations)

        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)

        plt.show()

    def run_test(self, plot_coords, agent1_type, agent2_type, alpha, epsilon, tau, gamma, episodes, simulations):

        a1_results = []
        a2_results = []

        for s in range(simulations):
            agent1 = agent1_type(alpha, epsilon, tau, gamma, len(self.actions), self.num_states)
            agent2 = agent2_type(alpha, epsilon, tau, gamma, len(self.actions), self.num_states)

            avg_reinforcement_values = self.run_simulation(agent1, agent2, episodes)

            a1_results.append(avg_reinforcement_values[0])
            a2_results.append(avg_reinforcement_values[1])

        a1_averages = []
        a2_averages = []
        for i in range(episodes):
            a1_val = 0
            a2_val = 0
            for j in range(simulations):
                a1_val += a1_results[j][i]
                a2_val += a2_results[j][i]
            a1_averages.append(a1_val / simulations)
            a2_averages.append(a2_val / simulations)

        # print(averages, '\n\n')
        self.axis[plot_coords[0], plot_coords[1]].plot([i for i in range(episodes)], a1_averages, label="agent 1")
        self.axis[plot_coords[0], plot_coords[1]].plot([i for i in range(episodes)], a2_averages, label="agent 2")

        # plt.ylim((-0.9, -0.3))
        self.axis[plot_coords[0], plot_coords[1]].set_xlabel("Episode Number")
        self.axis[plot_coords[0], plot_coords[1]].set_ylabel("Average Reinforcement Value")
        self.axis[plot_coords[0], plot_coords[1]].set_ylabel("Average Reinforcement Value")
        self.axis[plot_coords[0], plot_coords[1]].legend()
        self.axis[plot_coords[0], plot_coords[1]].set_ylim(self.ylim)
        self.axis[plot_coords[0], plot_coords[1]].set_title((agent1_type.get_name() + " & " + agent2_type.get_name()))


    def run_simulation(self, agent1: DoubleQLearning, agent2: DoubleQLearning, episodes):
        a1_avg_reinforcement_value = []
        a2_avg_reinforcement_value = []

        for _ in range(episodes):

            num_steps = 0
            a1_total = 0
            a2_total = 0
            # While neither agent has reached terminal
            while self.terminal_states.count(agent1.get_state()) == 0 and self.terminal_states.count(
                    agent2.get_state()) == 0:
                a1_action = agent1.get_action()
                a1_state = agent1.get_state()
                a2_action = agent2.get_action()
                a2_state = agent2.get_state()

                rewards = self.reward_function(a1_state, a1_action, a2_state, a2_action)

                agent1.reward(rewards[0], rewards[1])
                a1_total += rewards[0]
                agent2.reward(rewards[2], rewards[3])
                a2_total += rewards[2]

                num_steps += 1

            a1_avg_reinforcement_value.append(a1_total / num_steps)
            a2_avg_reinforcement_value.append(a2_total / num_steps)

            agent1.reset()
            agent2.reset()

        return a1_avg_reinforcement_value, a2_avg_reinforcement_value
