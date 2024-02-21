import numpy as np
from grid_world import GridWorld
import matplotlib.pyplot as plt
from policy import PolicyGrid
import math
import tqdm


class MDP:
    def __init__(self, grid_world, discount=1, prob_forward=1):
        self.grid_world = grid_world
        self.discount = discount
        self.prob_forward = prob_forward
        self.prob_sideways = (1.0 - prob_forward) / 2
        self.terminal_states = grid_world.get_terminal_states()
        self.optimal_values = np.zeros((grid_world.grid_size_x, grid_world.grid_size_y))
        self.infotogo = np.zeros((grid_world.grid_size_x, grid_world.grid_size_y, 4))
        self.delta = np.zeros((grid_world.grid_size_x, grid_world.grid_size_y, 4))
        self.z:dict[float, np.ndarray] = {}
        self.free_energy:dict[float, np.ndarray] = {}
        self.control_info:dict[float, np.ndarray] = {}
        self.policy_grid:dict[float, PolicyGrid] = {}
        self.value_grid = np.zeros((grid_world.grid_size_x, grid_world.grid_size_y))
        self.info_mdp_grid = np.zeros((grid_world.grid_size_x, grid_world.grid_size_y))

    def is_terminal(self, state):
        return state in self.terminal_states

    def get_next_states_and_probs(self, state):
        transitions = {}
        if self.is_terminal(state):
            return transitions

        actions = [
            ('u', (max(state[0] - 1, 0), state[1]), [('l', (state[0], max(state[1] - 1, 0))), ('r', (state[0], min(state[1] + 1, self.grid_world.grid_size_y - 1))) ]),
            ('d', (min(state[0] + 1, self.grid_world.grid_size_x - 1), state[1]), [('l', (state[0], max(state[1] - 1, 0))), ('r', (state[0], min(state[1] + 1, self.grid_world.grid_size_y - 1))) ]),
            ('l', (state[0], max(state[1] - 1, 0)), [('u', (max(state[0] - 1, 0), state[1])), ('d', (min(state[0] + 1, self.grid_world.grid_size_x - 1), state[1])) ]),
            ('r', (state[0], min(state[1] + 1, self.grid_world.grid_size_y - 1)), [('u', (max(state[0] - 1, 0), state[1])), ('d', (min(state[0] + 1, self.grid_world.grid_size_x - 1), state[1])) ])
        ]

        for action, new_state, sideways_states in actions:
            # Set default transitions for main actions to current state
            transitions[action] = [(state, self.prob_forward)]
            
            if self.grid_world.get_type(new_state) != 'O':
                transitions[action] = [(new_state, self.prob_forward)]
            
            # Add possible sideway transitions
            for sideway_action, sideway_state in sideways_states:
                if self.grid_world.get_type(sideway_state) != 'O':
                    transitions[action].append((sideway_state, self.prob_sideways))
                else:
                    transitions[action].append((state, self.prob_sideways))

        return transitions

    def evaluate_policy(self, policy_grid, theta=0.001, max_iterations=1000):
        """
        Perform policy evaluation.
        :param policy_grid: The policy to be evaluated.
        :param theta: A threshold of value change below which we consider value to have converged.
        :param max_iterations: Maximum number of iterations to prevent infinite loops.
        :return: None. This will update the value_grid matrix in-place.
        """
        self.value_grid = np.zeros((self.grid_world.grid_size_x, self.grid_world.grid_size_y))
        for iteration in range(max_iterations):
            delta = 0  # To store the change in value for stopping condition
            for x in range(self.grid_world.grid_size_x):
                for y in range(self.grid_world.grid_size_y):
                    state = (x, y)
                    v = self.value_grid[state]
                    new_v = 0

                    if not self.is_terminal(state):
                        reward = self.grid_world.get_reward(state)
                        for action, action_prob in policy_grid.action_probabilities(state).items():
                            for next_state, trans_prob in self.get_next_states_and_probs(state)[action]:
                                new_v += action_prob * trans_prob * (reward + self.discount * self.value_grid[next_state])

                    self.value_grid[state] = new_v
                    delta = max(delta, abs(v - new_v))
            
            if delta < theta:
                break

        return self.value_grid

    def value_iteration(self, threshold=1e-10, max_iter=10000):
        # for _ in tqdm.tqdm(range(max_iter)):
        for _ in range(max_iter):
            delta = 0
            for i in range(self.grid_world.grid_size_x):
                for j in range(self.grid_world.grid_size_y):
                    # Skip obstacle cells
                    if self.grid_world.get_type((i, j)) == 'O':
                        continue
                    if self.grid_world.get_type((i, j)) == 'T':
                        continue

                    old_value = self.optimal_values[i, j]
                    action_values = []

                    # Get all possible transitions for the current state
                    transitions = self.get_next_states_and_probs((i, j))
                    
                    r = self.grid_world.get_reward((i, j))
                    # print("Current state: ", (i, j), "Rwd: ", r)
                    # For each possible action, compute its expected value
                    for action, transition_list in transitions.items():
                        total_expected_value = 0
                        for next_state, prob in transition_list:
                            # print(action, next_state, prob)
                            if (action, next_state) == (action, transition_list[0][0]):
                                # This is the main forward transition
                                total_expected_value += r + self.discount * prob * self.optimal_values[next_state[0], next_state[1]]
                                # print("Main action:", action, "Next state value:", self.values[next_state[0], next_state[1]])
                            else:
                                # This is a sideways transition
                                sideway_value = self.discount * prob * self.optimal_values[next_state[0], next_state[1]]
                                # print("Side state:", next_state, "Sideway prob:", prob, "Side value:", sideway_value)
                                total_expected_value += sideway_value
                        action_values.append(total_expected_value)
                        # print("Total value for action", action, ":", total_expected_value)

                    # Choose the maximum expected action value for the state
                    if action_values:
                        self.optimal_values[i, j] = max(action_values)
                        # print("Max: ", max(action_values))
                        delta = max(delta, abs(self.optimal_values[i, j] - old_value))

            # Terminate when value changes are below the threshold
            if delta < threshold:
                break

    def info_to_go_iteration(self, prior_states, state_action_tans, state_action_prob, threshold=1, max_iter=1000, discount=0.75):
        # for _ in tqdm.tqdm(range(max_iter)):
        for _ in range(max_iter):
            delta = 0
            for i in range(self.grid_world.grid_size_x):
                for j in range(self.grid_world.grid_size_y):
                    # Skip obstacle cells
                    if self.grid_world.get_type((i, j)) == 'O':
                        continue
                    # if self.grid_world.get_type((i, j)) == 'T':
                    #     continue

                    for ia, action in enumerate(['u', 'd', 'l', 'r']):
                        # for neighbour in self.grid_world.neighbours((i,j)):
                        old_value = self.infotogo[i, j, ia]
                        action_values = []

                        # Get all possible transitions for the current state
                        neighbours = self.grid_world.neighbours((i,j))

                        # For each possible action, compute its expected value
                        total_expected_value = 0
                        d_val = 0
                        for neighbour in neighbours:
                            for ia1, action1 in enumerate(['u', 'd', 'l', 'r']):
                                sat = state_action_tans[(i,j)][action][neighbour]
                                sap = state_action_prob[neighbour][action1]
                                
                                prob = state_action_tans[(i,j)][action][neighbour]*state_action_prob[neighbour][action1]
                                neighbour_value = self.infotogo[i, j, ia1]
                                # print(state_action_tans[(i,j)][action][neighbour],prior_states[it+1][neighbour])
                                distance = self.grid_world.shortest_path_to_start(neighbour)
                                if distance == -1:
                                    distance = 0
                                else:
                                    distance = int(distance * self.prob_forward)
                                prior_states[distance][neighbour] = 0.25
                                val = math.log2((state_action_tans[(i,j)][action][neighbour]+0.000001)/(prior_states[distance][neighbour] + 0.000001))
                                val += math.log2((state_action_prob[neighbour][action1]+0.000001)/(state_action_prob[neighbour][action1]*prior_states[distance][neighbour] + 0.000001))  # prior_states[distance][neighbour]+0.000001))
                                d_val += prob * val
                                val += discount * neighbour_value
                                total_expected_value += prob * val
                                pass
                                # print("Total value for action", action, ":", total_expected_value)

                        self.infotogo[i, j, ia] = total_expected_value
                        self.delta[i, j, ia] = d_val
                        delta = max(delta, abs(total_expected_value - old_value))

            # Terminate when value changes are below the threshold
            if delta < threshold:
                break

    def info_rl(self, state_action_tans, beta, threshold=0, max_iter=1000):
        beta = np.float64(beta)
        threshold = np.float64(threshold)
        
        self.z[beta] = np.zeros((self.grid_world.grid_size_x, self.grid_world.grid_size_y), dtype=np.float64)
        self.free_energy[beta] = np.zeros((self.grid_world.grid_size_x, self.grid_world.grid_size_y), dtype=np.float64)
        self.control_info[beta] = np.zeros((self.grid_world.grid_size_x, self.grid_world.grid_size_y), dtype=np.float64)
        self.policy_grid[beta] = PolicyGrid(self.grid_world)
        
        # for _ in tqdm.tqdm(range(max_iter)):
        for _ in range(max_iter):
            delta = np.float64(0)
            for i in range(self.grid_world.grid_size_x):
                for j in range(self.grid_world.grid_size_y):
                    # Skip obstacle cells
                    if self.grid_world.get_type((i, j)) in ['O']:  # , 'T']:
                        continue
                    z = np.float64(0)
                    old_value = self.free_energy[beta][i, j]
                    for ia, action in enumerate(['u', 'd', 'l', 'r']):
                        reward = np.float64(self.grid_world.get_reward((i,j)))
                        # reward = 0
                        neighbour_free_energy = np.float64(0)
                        for k in range(self.grid_world.grid_size_x):
                            for m in range(self.grid_world.grid_size_y):
                                # reward += self.grid_world.get_reward((k,m)) * state_action_tans[(i,j)][action][(k,m)]
                                neighbour_free_energy += self.free_energy[beta][(k,m)] * state_action_tans[(i,j)][action][(k,m)]
                        prior_action_prob = np.float64(0.25)
                        z += prior_action_prob * math.exp(beta * reward - neighbour_free_energy)
                    self.z[beta][i, j] = z
                    _z = -np.log2(z)
                    self.free_energy[beta][i, j] = -np.log2(z)
                    delta = max(delta, abs(self.free_energy[beta][i, j] - old_value))
            if delta <= threshold:
                break

        for i in range(self.grid_world.grid_size_x):
            for j in range(self.grid_world.grid_size_y):
                # Skip obstacle cells
                if self.grid_world.get_type((i, j)) in ['O', 'T']:
                    continue
                for ia, action in enumerate(['u', 'd', 'l', 'r']):
                    reward = np.float64(self.grid_world.get_reward((i,j)))
                    # reward = 0
                    neighbour_free_energy = np.float64(0)
                    for k in range(self.grid_world.grid_size_x):
                        for m in range(self.grid_world.grid_size_y):
                            # reward += self.grid_world.get_reward((k,m)) * state_action_tans[(i,j)][action][(k,m)]
                            neighbour_free_energy += self.free_energy[beta][(k,m)] * state_action_tans[(i,j)][action][(k,m)]
                    prior_action_prob = np.float64(0.25)
                    prob = prior_action_prob * math.exp(beta * reward - neighbour_free_energy) / self.z[beta][i, j]
                    if action == 'u':
                        self.policy_grid[beta].policy_grid[i,j].u = prob
                    elif action == 'd':
                        self.policy_grid[beta].policy_grid[i,j].d = prob
                    elif action == 'l':
                        self.policy_grid[beta].policy_grid[i,j].l = prob
                    elif action == 'r':
                        self.policy_grid[beta].policy_grid[i,j].r = prob

        self.evaluate_policy(self.policy_grid[beta], beta)

        for i in range(self.grid_world.grid_size_x):
            for j in range(self.grid_world.grid_size_y):
                self.control_info[beta][i, j] = self.free_energy[beta][i, j] + beta * self.value_grid[beta][i, j]
                # print(self.control_info[beta][i, j], self.free_energy[beta][i, j], beta, self.value_grid[beta][i, j])
            
        return self.policy_grid[beta]

    def _info_rl(self, state_action_tans, beta, threshold=0, max_iter=1000):
        self.z[beta] = np.zeros((self.grid_world.grid_size_x, self.grid_world.grid_size_y))
        self.free_energy[beta] = np.zeros((self.grid_world.grid_size_x, self.grid_world.grid_size_y))
        self.control_info[beta] = np.zeros((self.grid_world.grid_size_x, self.grid_world.grid_size_y))
        self.policy_grid[beta] = PolicyGrid(self.grid_world)
        # for _ in tqdm.tqdm(range(max_iter)):
        for _ in range(max_iter):
            delta = 0
            for i in range(self.grid_world.grid_size_x):
                for j in range(self.grid_world.grid_size_y):
                    if i == j == 0:
                        pass
                    if i == 1 and j == 4:
                        pass
                    # Skip obstacle cells
                    if self.grid_world.get_type((i, j)) == 'O':
                        continue
                    if self.grid_world.get_type((i, j)) == 'T':
                        continue
                    z = 0
                    old_value = self.free_energy[beta][i, j]
                    for ia, action in enumerate(['u', 'd', 'l', 'r']):
                        # Get prior policy, state-action reward and neighbour free energy
                        # neighbours = self.grid_world.neighbours((i,j))
                        # free_directions = 0
                        reward = self.grid_world.get_reward((i,j))
                        neighbour_free_energy = 0
                        # for neighbour in neighbours:
                        # # reward += self.grid_world.get_reward()  # * state_action_tans[(i,j)][action][neighbour]
                        #     neighbour_free_energy += self.free_energy[beta][neighbour[0],neighbour[1]] * state_action_tans[(i,j)][action][neighbour]
                                # free_directions += 1
                        for k in range(self.grid_world.grid_size_x):
                            for m in range(self.grid_world.grid_size_y):
                                # reward += self.grid_world.get_reward()  # * state_action_tans[(i,j)][action][neighbour]
                                if state_action_tans[(i,j)][action][(k,m)] > 0:
                                    pass
                                _s = state_action_tans[(i,j)][action][(k,m)]
                                _f = self.free_energy[beta][(k,m)]
                                neighbour_free_energy += self.free_energy[beta][(k,m)] * state_action_tans[(i,j)][action][(k,m)]
                                # print((i,j), action, (k,m), state_action_tans[(i,j)][action][(k,m)], neighbour_free_energy)
                        prior_action_prob = 0.25
                        # if free_directions > 0:
                        #     prior_action_prob = 1 / free_directions
                        # try:
                        z += prior_action_prob * math.exp(beta * reward - neighbour_free_energy)
                        # print(neighbour_free_energy, math.exp(beta * reward - neighbour_free_energy))
                        # except OverflowError:
                        #     z += 0
                    if z <= 0:
                        z = 1e-250
                        # print("0000000000!!!!!!!")
                    self.z[beta][i, j] = z
                    _z = -math.log2(z)
                    # print(z, _z)
                    self.free_energy[beta][i, j] = -math.log2(z)
                    delta = max(delta, abs(self.free_energy[beta][i, j] - old_value))
                    # print(delta)

            # fig, ax = plt.subplots(figsize=(10, 10))
            # ax.set_xticks(np.arange(self.free_energy[beta].shape[1]))
            # ax.set_yticks(np.arange(self.free_energy[beta].shape[0]))
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])
            # ax.set_xticks(np.arange(self.free_energy[beta].shape[1]+1)-.5, minor=True)
            # ax.set_yticks(np.arange(self.free_energy[beta].shape[0]+1)-.5, minor=True)
            # ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
            # ax.tick_params(which="minor", size=0)
            # directions = ['u', 'd', 'l', 'r']
            # for k in range(self.free_energy[beta].shape[0]):
            #     for m in range(self.free_energy[beta].shape[1]):
            #         value = self.free_energy[beta][k, m].item()
            #         text = "%.2f" % value
            #         ax.text(m, k, text, ha='center', va='center', fontsize=10, color="black")
            # plt.gca().invert_yaxis()
            # plt.show()
            # pass

            # Terminate when value changes are below the threshold
            if delta <= threshold:
                break
        
        # fig, ax = plt.subplots(figsize=(10, 10))
        # ax.set_xticks(np.arange(self.free_energy[beta].shape[1]))
        # ax.set_yticks(np.arange(self.free_energy[beta].shape[0]))
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_xticks(np.arange(self.free_energy[beta].shape[1]+1)-.5, minor=True)
        # ax.set_yticks(np.arange(self.free_energy[beta].shape[0]+1)-.5, minor=True)
        # ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        # ax.tick_params(which="minor", size=0)
        # directions = ['u', 'd', 'l', 'r']
        # for k in range(self.free_energy[beta].shape[0]):
        #     for m in range(self.free_energy[beta].shape[1]):
        #         value = self.free_energy[beta][k, m].item()
        #         text = "%.2f" % value
        #         ax.text(m, k, text, ha='center', va='center', fontsize=10, color="black")
        # plt.gca().invert_yaxis()
        # plt.show()

        for i in range(self.grid_world.grid_size_x):
            for j in range(self.grid_world.grid_size_y):
                _c = self.free_energy[beta][i, j]
                _d = beta * self.optimal_values[i, j]
                self.control_info[beta][i, j] = self.free_energy[beta][i, j] + beta * self.optimal_values[i, j]
                # Skip obstacle cells
                if self.grid_world.get_type((i, j)) == 'O':
                    continue
                if self.grid_world.get_type((i, j)) == 'T':
                    continue
                for ia, action in enumerate(['u', 'd', 'l', 'r']):
                    # neighbours = self.grid_world.neighbours((i,j))
                    # free_directions = 0
                    reward = self.grid_world.get_reward((i,j))
                    neighbour_free_energy = 0
                    for k in range(self.grid_world.grid_size_x):
                        for m in range(self.grid_world.grid_size_y):
                            # reward += self.grid_world.get_reward()  # * state_action_tans[(i,j)][action][neighbour]
                            neighbour_free_energy += self.free_energy[beta][(k,m)] * state_action_tans[(i,j)][action][(k,m)]
                            _t = state_action_tans[(i,j)][action][(k,m)]
                            # print((i,j), action, (k,m), _t, neighbour_free_energy)
                            # free_directions += 1
                    prior_action_prob = 0.25
                    # if free_directions > 0:
                    #     prior_action_prob = 1 / free_directions
                    if action == 'u':
                        self.policy_grid[beta].policy_grid[i,j].u = prior_action_prob * math.exp(beta * reward - neighbour_free_energy) / self.z[beta][i, j]
                        _a0 = beta * reward - neighbour_free_energy
                        _b0 = prior_action_prob * math.exp(beta * reward - neighbour_free_energy)
                        z = self.z[beta][i, j]
                        u = self.policy_grid[beta].policy_grid[i,j].u
                        pass
                    if action == 'd':
                        self.policy_grid[beta].policy_grid[i,j].d = prior_action_prob * math.exp(beta * reward - neighbour_free_energy) / self.z[beta][i, j]
                        _a1 = beta * reward - neighbour_free_energy
                        _b1 = prior_action_prob * math.exp(beta * reward - neighbour_free_energy)
                        z = self.z[beta][i, j]
                        d = self.policy_grid[beta].policy_grid[i,j].d
                        pass
                    if action == 'l':
                        self.policy_grid[beta].policy_grid[i,j].l = prior_action_prob * math.exp(beta * reward - neighbour_free_energy) / self.z[beta][i, j]
                        _a2 = beta * reward - neighbour_free_energy
                        _b2 = prior_action_prob * math.exp(beta * reward - neighbour_free_energy)
                        z = self.z[beta][i, j]
                        l = self.policy_grid[beta].policy_grid[i,j].l
                        pass
                    if action == 'r':
                        self.policy_grid[beta].policy_grid[i,j].r = prior_action_prob * math.exp(beta * reward - neighbour_free_energy) / self.z[beta][i, j]
                        _a3 = beta * reward - neighbour_free_energy
                        _b3 = prior_action_prob * math.exp(beta * reward - neighbour_free_energy)
                        z = self.z[beta][i, j]
                        r = self.policy_grid[beta].policy_grid[i,j].r
                        pass
                    try:
                        s = _b0 + _b1 + _b2 + _b3
                        s1 = (_b0 + _b1 + _b2 + _b3) / 4
                        pass
                    except Exception as e:
                        pass
        
        return self.policy_grid[beta]

    def info_mdp_with_prior(self, state_action_tans, policy: PolicyGrid, prior_policy: PolicyGrid, gamma=1.0, threshold=0, max_iter=1000):
        # for _ in tqdm.tqdm(range(max_iter)):
        for _ in range(max_iter):
            delta = 0
            for i in range(self.grid_world.grid_size_x):
                for j in range(self.grid_world.grid_size_y):
                    old_value = self.info_mdp_grid[i,j]

                    # Skip obstacle cells
                    if self.grid_world.get_type((i, j)) in ['O', 'T']:
                        continue
                    
                    neighbours = self.grid_world.neighbours((i,j)) + [(i,j)]
                    state_action_probs = policy.action_probabilities((i,j))
                    prior_station_action_probs = prior_policy.action_probabilities((i,j))
                    
                    delta_i = 0
                    for a in ['u', 'd', 'r', 'l']:
                        if state_action_probs[a] <= 0:
                            continue
                        # l = math.log2(state_action_probs[a]/0.25)
                        delta_i += state_action_probs[a] * math.log2(state_action_probs[a]/(prior_station_action_probs[a]+0.000001))

                    neighbour_value = 0
                    for neighbour in neighbours:
                        # get transition probability
                        prob_trans = 0
                        for a in ['u', 'd', 'r', 'l']:
                            prob_trans += state_action_probs[a] * state_action_tans[(i,j)][a][neighbour]
                        
                        neighbour_value += prob_trans * self.info_mdp_grid[neighbour[0],neighbour[1]]
                    
                    new_value = delta_i + gamma * neighbour_value
                    self.info_mdp_grid[i, j] = new_value
                    delta = max(delta, abs(self.info_mdp_grid[i, j] - old_value))
            if delta <= threshold:
                break

        return self.info_mdp_grid
    
    def info_mdp(self, state_action_tans, policy: PolicyGrid, gamma=1.0, threshold=0, max_iter=1000):
        # for _ in tqdm.tqdm(range(max_iter)):
        for _ in range(max_iter):
            delta = 0
            for i in range(self.grid_world.grid_size_x):
                for j in range(self.grid_world.grid_size_y):
                    old_value = self.info_mdp_grid[i,j]

                    # Skip obstacle cells
                    if self.grid_world.get_type((i, j)) in ['O', 'T']:
                        continue
                    
                    neighbours = self.grid_world.neighbours((i,j)) + [(i,j)]
                    state_action_probs = policy.action_probabilities((i,j))
                    
                    delta_i = 0
                    for a in ['u', 'd', 'r', 'l']:
                        if state_action_probs[a] <= 0:
                            continue
                        # l = math.log2(state_action_probs[a]/0.25)
                        delta_i += state_action_probs[a] * math.log2(state_action_probs[a]/0.25)

                    neighbour_value = 0
                    for neighbour in neighbours:
                        # get transition probability
                        prob_trans = 0
                        for a in ['u', 'd', 'r', 'l']:
                            prob_trans += state_action_probs[a] * state_action_tans[(i,j)][a][neighbour]
                        
                        neighbour_value += prob_trans * self.info_mdp_grid[neighbour[0],neighbour[1]]
                    
                    new_value = delta_i + gamma * neighbour_value
                    self.info_mdp_grid[i, j] = new_value
                    delta = max(delta, abs(self.info_mdp_grid[i, j] - old_value))
            if delta <= threshold:
                break

        return self.info_mdp_grid

    def print_values(self):
        print("State Values:")
        for i in range(self.grid_world.grid_size_x):
            for j in range(self.grid_world.grid_size_y):
                print(f"{self.optimal_values[i, j]:.2f}", end="\t")
            print()

    def visualize_values(self):
        fig, ax = plt.subplots()
        cax = ax.matshow(self.optimal_values, cmap='viridis')
        # fig.colorbar(cax)
        for i in range(self.grid_world.grid_size_x):
            for j in range(self.grid_world.grid_size_y):
                ax.text(j, i, f"{self.optimal_values[i, j]:.2f}", ha='center', va='center', color='w')
        # plt.show()
    
    def visualize_info(self):
        fig, ax = plt.subplots()
        cax = ax.matshow(self.info_mdp_grid, cmap='viridis')
        # fig.colorbar(cax)
        for i in range(self.grid_world.grid_size_x):
            for j in range(self.grid_world.grid_size_y):
                ax.text(j, i, f"{self.info_mdp_grid[i, j]:.2f}", ha='center', va='center', color='w')

    def get_optimal_policy(self, prob_forward=1, threshold=1e-3):
        prob_elsewhere = (1.0 - prob_forward) / 3
        optimal_policy_grid = PolicyGrid(self.grid_world)

        for i in range(self.grid_world.grid_size_x):
            for j in range(self.grid_world.grid_size_y):
                if self.grid_world.get_type((i, j)) in ['O', 'T']:
                    continue

                transitions = self.get_next_states_and_probs((i, j))
                action_values = {}

                for action, transition_list in transitions.items():
                    total_expected_value = 0
                    for next_state, prob in transition_list:
                        total_expected_value += prob * self.optimal_values[next_state[0], next_state[1]]
                    action_values[action] = total_expected_value

                max_value = max(action_values.values())
                optimal_actions = [action for action, value in action_values.items() if max_value - value <= threshold]

                current_policy = optimal_policy_grid.get_policy((i, j))
                for action in ['u', 'd', 'l', 'r']:
                    if action in optimal_actions:
                        current_policy.__dict__[action] = prob_forward / len(optimal_actions)
                    else:
                        current_policy.__dict__[action] = prob_elsewhere

        return optimal_policy_grid

    
    # def get_balance_info_policy(self, forward=1, elsewhere=0):
    #     optimal_policy_grid = PolicyGrid(self.grid_world)

    #     for i in range(self.grid_world.grid_size_x):
    #         for j in range(self.grid_world.grid_size_y):
    #             if self.grid_world.get_type((i, j)) in ['O', 'T']:
    #                 continue

    #             transitions = self.get_next_states_and_probs((i, j))
    #             action_values = {}

    #             for action, transition_list in transitions.items():
    #                 total_expected_value = 0
    #                 for next_state, prob in transition_list:
    #                     total_expected_value += prob * self.values[next_state[0], next_state[1]]
    #                 action_values[action] = total_expected_value

    #             best_action = max(action_values, key=action_values.get)
                
    #             current_policy = optimal_policy_grid.get_policy((i, j))
    #             current_policy.u = forward if best_action == 'u' else elsewhere
    #             current_policy.d = forward if best_action == 'd' else elsewhere
    #             current_policy.l = forward if best_action == 'l' else elsewhere
    #             current_policy.r = forward if best_action == 'r' else elsewhere

    #     return optimal_policy_grid



if __name__ == "__main__":
    layout_file = "map.txt"

    env = GridWorld(layout_file)
    mdp = MDP(env, discount=0.99, prob_forward=0.8, prob_sideways=0.1)

    mdp.value_iteration()

    env.visualize_grid()
    mdp.visualize_values()

    optimal_policy_grid = mdp.get_optimal_policy()
    optimal_policy_grid.visualize()
    plt.show()
