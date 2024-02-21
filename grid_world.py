import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import deque

class GridWorld:
    def __init__(self, layout_file):
        self.rewards_map = {
            'T': 0.0,  # Target
            'S': -1.0,   # Starting point
            'O': -1.0,  # Obstacle
            '+': 5.0,   # Treasure
            'X': -20.0,  # Trap
            '.': -1.0   # Empty cell
        }
        self.layout = self._load_layout(layout_file)
        self.grid_size_x, self.grid_size_y = self.layout.shape

    def _load_layout(self, filename):
        with open(filename, 'r') as f:
            layout = [list(line.strip()) for line in f.readlines()]
        return np.array(layout)

    def get_reward(self, state):
        cell_type = self.layout[state[0], state[1]]
        return self.rewards_map[cell_type]

    def get_type(self, state):
        return self.layout[state[0], state[1]]

    def find_positions(self, cell_type):
        return [(i, j) for i in range(self.grid_size_x) for j in range(self.grid_size_y) if self.layout[i, j] == cell_type]

    def get_terminal_states(self):
        return self.find_positions('T')
    
    def neighbours(self, state: tuple) -> list[tuple]:
        """Given a state, return its accessible neighbouring states."""
        
        x, y = state
        possible_neighbours = [
            (x - 1, y),  # Up
            (x + 1, y),  # Down
            (x, y - 1),  # Left
            (x, y + 1)   # Right
        ]
        
        valid_neighbours = []
        for neighbour in possible_neighbours:
            nx, ny = neighbour
            # Check boundary conditions and whether it's an obstacle or target
            if 0 <= nx < self.grid_size_x and 0 <= ny < self.grid_size_y and \
               self.get_type(neighbour) not in ['T', 'O']:
                valid_neighbours.append(neighbour)
                
        return valid_neighbours
    
    def shortest_path_to_start(self, state: tuple) -> int:
        """Return the shortest path length from the given state to the start state using BFS."""

        start_states = self.find_positions('S')
        if not start_states:  # If there are no starting positions, return -1
            return -1

        # Use BFS to find the shortest path
        visited = set()
        queue = deque([(state, 0)])  # Each element is (current_state, path_length)

        while queue:
            current_state, path_length = queue.popleft()

            if current_state in start_states:
                return path_length  # Return the path length when a start state is reached

            visited.add(current_state)

            # Add neighbouring states to the queue
            for neighbour in self.neighbours(current_state):
                if neighbour not in visited:
                    queue.append((neighbour, path_length + 1))

        return -1  # If no path is found, return -1
    
    def get_state_action_transitions(self, success_prob):
        slip_prob = (1.0 - success_prob) / 2
        state_action_trans = {}

        for x in range(self.grid_size_x):
            for y in range(self.grid_size_y):
                state_action_trans[(x, y)] = {}

                for action in ['u', 'd', 'l', 'r']:
                    state_action_trans[(x, y)][action] = {}

                    # Initialize transitions to zero for all states
                    for nx in range(self.grid_size_x):
                        for ny in range(self.grid_size_y):
                            state_action_trans[(x, y)][action][(nx, ny)] = 0

                    # Calculate the outcome state for each action and adjust probabilities
                    for possible_action in ['u', 'd', 'l', 'r']:
                        transition_state = self._get_transition_state((x, y), possible_action)

                        # Assign transition probabilities based on intended action and possible action
                        if possible_action == action:
                            # Intended action
                            state_action_trans[(x, y)][action][transition_state] = success_prob
                        else:
                            # Slip actions
                            if (action in ['u', 'd'] and possible_action in ['l', 'r']) or (action in ['l', 'r'] and possible_action in ['u', 'd']):
                                # Slip to a side
                                state_action_trans[(x, y)][action][transition_state] += slip_prob

        return state_action_trans

    def _get_transition_state(self, state, action):
        # Attempt to move in the specified direction
        transitions = {
            'u': (state[0] - 1, state[1]),
            'd': (state[0] + 1, state[1]),
            'l': (state[0], state[1] - 1),
            'r': (state[0], state[1] + 1)
        }
        transition_state = transitions[action]
        
        # Check for boundary conditions
        if not (0 <= transition_state[0] < self.grid_size_x and 0 <= transition_state[1] < self.grid_size_y):
            return state  # Stay in the same state if it's out of bounds

        # Check for walls or obstacles
        if self.get_type(transition_state) in ['O']:  # , 'T'
            return state  # Stay in the same state if there's an obstacle or it's a terminal stat
        
        # Absorb state
        if self.get_type(state) in ['T']:
            return state

        return transition_state


    def _get_side_transitions(self, action):
        # Get the side actions based on the current action
        side_actions = {
            'u': ['l', 'r'],
            'd': ['l', 'r'],
            'l': ['u', 'd'],
            'r': ['u', 'd']
        }
        return side_actions[action]

    def print_grid(self):
        print("Grid World:")
        for row in self.layout:
            print("\t".join(row))
            print()

    def print_rewards(self):
        print("State Rewards:")
        for i in range(self.grid_size_x):
            for j in range(self.grid_size_y):
                print(f"{self.get_reward((i, j)):.2f}", end="\t")
            print()

    def visualize_grid(self):
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
        data = np.zeros(self.layout.shape, dtype=int)
        for i in range(self.grid_size_x):
            for j in range(self.grid_size_y):
                cell_type = self.layout[i, j]
                data[i, j] = cell_types_num[cell_type]

        ax.imshow(data, cmap=cmap)

        # Display rewards on the grid
        for i in range(self.grid_size_x):
            for j in range(self.grid_size_y):
                ax.text(j, i, str(self.get_reward((i, j))), ha='center', va='center', color='w')

        # plt.show()

