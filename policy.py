from grid_world import GridWorld
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


class Policy:
    def __init__(self, u=0.0, d=0.0, l=0.0, r=0.0) -> None:
        self.u = u
        self.d = d
        self.l = l
        self.r = r

    def to_list(self):
        """Convert the policy to a list."""
        return [self.u, self.d, self.l, self.r]

    def update_from_list(self, values):
        """Update the policy from a list of values."""
        self.u, self.d, self.l, self.r = values

    def __repr__(self):
        return f"Policy(u={self.u}, d={self.d}, l={self.l}, r={self.r})"


class PolicyGrid:
    def __init__(self, grid_world: GridWorld, default=(0, 0, 0, 0)) -> None:
        self.policy_grid:dict[tuple, Policy] = {}
        self.grid_world = grid_world
        for x in range(self.grid_world.grid_size_x):
            for y in range(self.grid_world.grid_size_y):
                self.policy_grid[(x,y)] = Policy(*default)
    
    def get_policy(self, state: tuple):
        return self.policy_grid[state]
    
    def action_probabilities(self, state: tuple) -> dict[str, float]:
        """Given a state, return the probabilities of taking each action."""
        
        policy_at_state = self.get_policy(state)
        return {
            'u': policy_at_state.u,
            'd': policy_at_state.d,
            'l': policy_at_state.l,
            'r': policy_at_state.r
        }
    
    def interpolate(self, gamma):
        """Create a new PolicyGrid with interpolated policies based on gamma."""
        new_policy_grid = PolicyGrid(self.grid_world)
        num_actions = 4  # Four possible actions: up, down, left, right

        for state, policy in self.policy_grid.items():
            original_policy_list = policy.to_list()
            interpolated_policy_list = [gamma * p + (1 - gamma) * (1 / num_actions) for p in original_policy_list]
            new_policy_grid.policy_grid[state].update_from_list(interpolated_policy_list)

        return new_policy_grid
    
    def visualize(self):
        fig, ax = plt.subplots()

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
        data = np.zeros(self.grid_world.layout.shape, dtype=int)
        for i in range(self.grid_world.grid_size_x):
            for j in range(self.grid_world.grid_size_y):
                cell_type = self.grid_world.layout[i, j]
                data[i, j] = cell_types_num[cell_type]

        ax.imshow(data, cmap=cmap)

        # Display policy as arrows on the grid
        for state, policy in self.policy_grid.items():
            i, j = state
            if policy.r != 0:
                ax.arrow(j, i, 0.3 * policy.r, 0, head_width=0.1, head_length=0.1, fc='white', ec='white')
            if policy.l != 0:
                ax.arrow(j, i, -0.3 * policy.l, 0, head_width=0.1, head_length=0.1, fc='white', ec='white')
            if policy.u != 0:
                ax.arrow(j, i, 0, -0.3 * policy.u, head_width=0.1, head_length=0.1, fc='white', ec='white')
            if policy.d != 0:
                ax.arrow(j, i, 0, 0.3 * policy.d, head_width=0.1, head_length=0.1, fc='white', ec='white')

        # plt.show()


if __name__ == "__main__":
    # Assume you have an instance of GridWorld as gridworld
    layout_file = "map.txt"
    gridworld = GridWorld(layout_file)
    policy_grid = PolicyGrid(grid_world=gridworld)
    policy_grid.visualize()
