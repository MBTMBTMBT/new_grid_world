import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from grid_world import GridWorld
from policy import PolicyGrid
from simulator import Simulator
# from mdp import MDP

class Sampler:
    def __init__(self, grid_world: GridWorld, prob_forward: float, prob_sideways: float) -> None:
        self.grid_world = grid_world
        self.simulator = Simulator(self.grid_world, prob_forward, prob_sideways)
    
    def sample(self, policy: PolicyGrid, iterations: int, steps: int) -> tuple[dict, dict]:
        count_states = {}
        rewards = {}
        for j in range(steps):
            count_states[j] = []
            rewards[j] = []
        for i in range(iterations):
            self.simulator.start()
            # print(self.simulator.current)
            count_states[0].append(self.simulator.current)
            for j in range(steps):
                if self.simulator.end():
                    break
                self.simulator.step(policy)
                count_states[j].append(self.simulator.current)
                rewards[j].append(self.simulator.total_reward)
                # print(i, self.simulator.current)
            # print("Reward: ", self.simulator.total_reward)
        return count_states, rewards
    
def state_probability_at_time_t(t: int, state: tuple, count_states: dict):
    iterations = len(count_states[0])
    if t not in count_states:
        return 0
    count = count_states[t].count(state)
    return count / iterations

def visualize_state_probability_at_time_t(grid_world: GridWorld, count_states: dict, t: int):
    # Define colors for each cell type
    cell_colors = {
        '.': 'grey',
        'T': 'yellow',
        'S': 'blue',
        'O': 'black',
        '+': 'green',
        'X': 'red'
    }

    # Convert the cell type to a unique number for coloring
    cell_types_num = {
        '.': 0,
        'T': 1,
        'S': 2,
        'O': 3,
        '+': 4,
        'X': 5
    }
    
    cmap = mcolors.ListedColormap(list(cell_colors.values()))

    # Convert layout to numerical values for coloring
    data = np.zeros(grid_world.layout.shape, dtype=int)
    for i in range(grid_world.grid_size_x):
        for j in range(grid_world.grid_size_y):
            cell_type = grid_world.layout[i, j]
            data[i, j] = cell_types_num[cell_type]

    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap)

    # Display the probability values on each cell
    for i in range(grid_world.grid_size_x):
        for j in range(grid_world.grid_size_y):
            state = (i, j)
            probability = state_probability_at_time_t(t, state, count_states)
            ax.text(j, i, f"{probability:.2f}", ha='center', va='center', color='w', fontsize=10)

    plt.title(f"State Probabilities at Time {t}")

if __name__ == "__main__":
    from mdp import MDP
    layout_file = "map.txt"
    env = GridWorld(layout_file)
    rand_policy = PolicyGrid(env, (0.25, 0.25, 0.25, 0.25))
    mdp = MDP(env, discount=0.99, prob_forward=0.8, prob_sideways=0.1)
    mdp.value_iteration()
    mdp_policy = mdp.get_optimal_policy()
    sampler = Sampler(env, 0.8, 0.1)

    states_rand, rwd_rand = sampler.sample(rand_policy, 100, 1000)
    visualize_state_probability_at_time_t(env, states_rand, 6)

    states_rand, rwd_rand = sampler.sample(mdp_policy, 100, 1000)
    visualize_state_probability_at_time_t(env, states_rand, 6)
    plt.show()
