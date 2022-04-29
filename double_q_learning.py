import random
import matplotlib.pyplot as plt

class DoubleQLearning:
    def softmax_action_selection(self, state, tau):
        rand = random.random()
        e = 2.71828
        action_probs = []
        denom = 0
        for action in range(len(self.actions)):
            denom += e**(self.merged_q_table[state][action]/tau)
        for action in range(len(self.actions)):
            num = e**(self.merged_q_table[state][action]/tau)
            action_probs[action] = num / denom
        
        chosen_action = 0
        while rand > 0:
            rand -= action_probs[chosen_action]
            chosen_action += 1
            
        return chosen_action

    # Select actions using epsilon greedy while handling the exploration-exploitation dilemma
    def egreedy_action_selection(self, state, epsilon):
        # Generate random number between 0 and 1
        rand = random.random()
        if rand < epsilon:
            return random.randint(0, len(self.merged_q_table[state])-1)
        max_value = max(self.merged_q_table[state])
        return self.merged_q_table[state].index(max_value)

    # def take_action(self, state, action):
    #     new_row = state[0] + self.actions[action][0]
    #     new_col = state[1] + self.actions[action][1]
    #     if new_row < 0 or new_row >= 4 or new_col < 0 or new_col >= 4:
    #         return (self.reward[state[0]][state[1]], state)
    #     return (self.reward[new_row][new_col], (new_row, new_col))

    def double_q_learning(self, alpha, epsilon, gamma, episodes, simulations):
        # Initialize 16x4 matrix (4 actions for each of the 16 states)
        # Action 1 = up
        # Action 2 = down
        # Action 3 = left
        # Action 4 = right
        results = []
        # Initialize reward table
        self.reward = [[11, -30, 0], [-30, 7, 6], [0, 0, 5]]

        # self.reward = [[-1] * 4 for _ in range(4)]
        # self.reward[0][0] = 1
        # self.reward[3][3] = 1

        # self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for _ in range(simulations):
            # self.q_table = [[0] * 4 for _ in range(16)]
            self.q_table_a = [[0,0,0]]
            self.q_table_b = [[0,0,0]]
            self.merged_q_table = []
            avg_reinf_value = []
            time = 0
            for _ in range(episodes):
                total = 0
                time += 1
                self.merged_q_table = [[(x+y)/2 for x, y in zip(self.q_table_a[0], self.q_table_b[0])]]
                # Choose an action using e-greedy method and take the action and store Reward and next state
                if epsilon == "1/T":
                    action = self.egreedy_action_selection(0, 1/time)
                else:
                    action = self.egreedy_action_selection(0, epsilon)
                # action = self.egreedy_action_selection(0, epsilon)
                
                # Opponent's action (random agent)
                # opp_action = random.randint(0, 2)
                # opp_action = random.randint(0, 2)
                opp_action = 0

                reward = self.reward[action][opp_action]
                rand_int = random.randint(0, 1)
                # Update Q-table A
                if rand_int == 1:
                    a_max = max(self.q_table_a[0])
                    a_act = self.q_table_a[0].index(a_max)
                    self.q_table_a[0][action] = self.q_table_a[0][action] + alpha * (reward + (gamma * self.q_table_b[0][a_act]) - self.q_table_a[0][action])
                else:
                    b_max = max(self.q_table_b[0])
                    b_act = self.q_table_b[0].index(b_max)
                    self.q_table_b[0][action] = self.q_table_b[0][action] + alpha * (reward + (gamma * self.q_table_a[0][b_act]) - self.q_table_b[0][action])
                # reward, next_state = self.take_action(cur_state, action)
                
                # total += reward
                # avg_reinf_value.append(total/num_steps)
                avg_reinf_value.append(reward)
            results.append(avg_reinf_value)
        averages = []
        for i in range(episodes):
            val = 0
            for j in range(simulations):
                val += results[j][i]
            averages.append(val/simulations)
        # print(averages, '\n\n')
        plt.plot([i for i in range(episodes)], averages)
        plt.plot([i for i in range(episodes)], [-1/3 for _ in range(episodes)])
        # plt.ylim((-0.9, -0.3))
        plt.xlabel("Episode Number")
        plt.ylabel("Average Reinforcement Value")
        plt.show()
                

test = DoubleQLearning()
test.double_q_learning(0.1, "1/T", 0.9, 500, 10)
# test.double_q_learning(1, 1, 1, 500, 10)

# test.q_learning(0.1, 0.25, 0.9, 1000, 10)
# test.q_learning("1/T", "1/T", 0.9, 1000, 10)
# test.q_learning(0.1, "1/T", 0.9, 1000, 10)
# test.q_learning("1/T", 0.1, 0.9, 1000, 10)
# test.q_learning(1, 1, 1, 1000, 10)

# test.q_learning(0.1, 0.1, 0.9, 1000, 10)