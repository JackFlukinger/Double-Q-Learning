import matplotlib.pyplot as plt

from q_learning.double_q_learning import DoubleQLearning
from q_learning.double_q_learning_egreedy import DoubleQLearningEGreedy
from q_learning.double_q_learning_random import DoubleQLearningRandom
from q_learning.double_q_learning_softmax import DoubleQLearningSoftmax


class EnvironmentInterface:
    actions = []
    terminal_states = []
    ylim = [-50, 50]
    num_states = 0
    figure, axis = plt.subplots(2, 2)

    def reward_function(self, p1_state: int, p1_action: int, p2_state: int, p2_action: int) -> (int, int, int, int):
        """Return tuple (p1_reward, p1_state, p2_reward, p2_state) for agents taking actions"""
        pass

    def run_tests(self, alpha, epsilon, tau, delta, gamma, episodes, simulations=1):
        self.run_test((0, 0), DoubleQLearningSoftmax, DoubleQLearningSoftmax, alpha, epsilon, tau, delta, gamma, episodes,
                      simulations)
        self.run_test((0, 1), DoubleQLearningSoftmax, DoubleQLearningRandom, alpha, epsilon, tau, delta, gamma, episodes,
                      simulations)
        self.run_test((1, 0), DoubleQLearningEGreedy, DoubleQLearningEGreedy, alpha, epsilon, tau, delta, gamma, episodes,
                      simulations)
        self.run_test((1, 1), DoubleQLearningEGreedy, DoubleQLearningRandom, alpha, epsilon, tau, delta, gamma, episodes,
                      simulations)

        self.figure.set_size_inches(10.5, 6.5)

        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.85,
                            wspace=0.6,
                            hspace=0.6)

        plt.show()

    def run_test(self, plot_coords, agent1_type, agent2_type, alpha, epsilon, tau, delta, gamma, episodes, simulations):

        a1_results = []
        a2_results = []
        optimal_rewards = []

        # track optimal policies
        a1_op_results = [0] * self.num_states
        a1_op_reward = 0
        a2_op_results = [0] * self.num_states
        a2_op_reward = 0

        for s in range(simulations):
            agent1 = agent1_type(alpha, epsilon, tau, delta, gamma, len(self.actions), self.num_states)
            agent2 = agent2_type(alpha, epsilon, tau, delta, gamma, len(self.actions), self.num_states)

            results = self.run_simulation(agent1, agent2, episodes)

            a1_results.append(results[0])
            a2_results.append(results[1])
            optimal_rewards.append(results[2])

            op_results = self.run_optimal_policy(agent1, agent2)
            a1_op_results[op_results[1]] += 1
            a2_op_results[op_results[3]] += 1
            a1_op_reward += op_results[0]
            a2_op_reward += op_results[2]

        print(agent1_type.get_name() + " & " + agent2_type.get_name() + ": ")
        print("    Agent 1:")
        for i in range(0, len(a1_op_results)):
            if a1_op_results[i] != 0:
                print("        State", i, ":", (a1_op_results[i] / simulations) * 100, "%")
        print("        Average Reward:", (a1_op_reward / simulations))

        print("    Agent 2:")
        for i in range(0, len(a2_op_results)):
            if a2_op_results[i] != 0:
                print("        State", i, ":", (a2_op_results[i] / simulations) * 100, "%")
        print("        Average Reward:", (a2_op_reward / simulations))

        a1_averages = []
        a2_averages = []
        op_reward_averages = []
        for i in range(episodes):
            a1_val = 0
            a2_val = 0
            op_val = 0
            for j in range(simulations):
                a1_val += a1_results[j][i]
                a2_val += a2_results[j][i]
                op_val += optimal_rewards[j][i]
            a1_averages.append(a1_val / simulations)
            a2_averages.append(a2_val / simulations)
            op_reward_averages.append(op_val / simulations)

        self.figure.suptitle(self.name)
        self.axis[plot_coords[0], plot_coords[1]].plot([i for i in range(episodes)], a1_averages, label="avg reinforcement")
        self.axis[plot_coords[0], plot_coords[1]].plot([i for i in range(episodes)], op_reward_averages, color='red', label="agent policy")
        self.axis[plot_coords[0], plot_coords[1]].axhline(y=self.optimal, color='orange', label="optimal policy")

        self.axis[plot_coords[0], plot_coords[1]].set_xlabel("Episode Number")
        self.axis[plot_coords[0], plot_coords[1]].set_ylabel("Reward")
        self.axis[plot_coords[0], plot_coords[1]].legend()
        self.axis[plot_coords[0], plot_coords[1]].set_ylim(self.ylim)
        self.axis[plot_coords[0], plot_coords[1]].set_title((agent1_type.get_name() + " & " + agent2_type.get_name()))

    def run_simulation(self, agent1: DoubleQLearning, agent2: DoubleQLearning, episodes):
        a1_avg_reinforcement_value = []
        a2_avg_reinforcement_value = []
        optimal_rewards = []

        for _ in range(episodes):
            #print("Episode", _)
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

            optimal_rewards.append(self.run_optimal_policy(agent1, agent2)[0])

            agent1.reset_state()
            agent2.reset_state()

        agent1.reset_time()
        agent2.reset_time()

        return a1_avg_reinforcement_value, a2_avg_reinforcement_value, optimal_rewards

    def run_optimal_policy(self, agent1: DoubleQLearning, agent2: DoubleQLearning, start_state=0):
        a1_cur_state = start_state
        a2_cur_state = start_state
        a1_reward = 0
        a2_reward = 0
        while self.terminal_states.count(a1_cur_state) == 0 and self.terminal_states.count(a2_cur_state) == 0:

            a1_optimal_action = agent1.optimal_action(a1_cur_state)
            a2_optimal_action = agent2.optimal_action(a2_cur_state)
            result = self.reward_function(a1_cur_state, a1_optimal_action, a2_cur_state,
                                          a2_optimal_action)
            a1_cur_state = result[1]
            a1_reward += result[0]
            a2_cur_state = result[3]
            a2_reward += result[2]
        return a1_reward, a1_cur_state, a2_reward, a2_cur_state
