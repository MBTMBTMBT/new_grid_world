from grid_world import GridWorld
from policy import PolicyGrid
import random


class Simulator:
    def __init__(self, grid_world: GridWorld, prob_forward=1.0, prob_sideways=0.0) -> None:  # policy: policy.PolicyGrid
        self.grid_world = grid_world
        self.prob_forward = prob_forward
        self.prob_sideways = prob_sideways
        # self.policy = policy
        self.starts = self.grid_world.find_positions('S')
        self.ends = self.grid_world.find_positions('T')
        self.obstacles = self.grid_world.find_positions('O')
        
        self.total_reward = 0
        self.current= ()

    def start(self, starting_point=None) -> None:
        self.total_reward = 0

        # select a starting point
        if starting_point is not None:
            self.current = starting_point
        else:
            if len(self.starts) == 0:
                for _ in range(50):
                    self.current = (random.randint(0,self.grid_world.grid_size_x-1), (random.randint(0,self.grid_world.grid_size_y-1)))
                    if not self.end():
                        break
            else:
                self.current = self.starts[random.randint(0, len(self.starts)-1)]
    
    def end(self) -> bool:
        if len(self.ends) == 0 and len(self.obstacles) == 0:
            return False
        else:
            return self.current in self.ends or self.current in self.obstacles
        
    def step(self, policy: PolicyGrid) -> tuple:
        policy_at_current = policy.get_policy(self.current)
        actions = ['u', 'd', 'l', 'r']
        probabilities = [policy_at_current.u, policy_at_current.d, policy_at_current.l, policy_at_current.r]
        if sum(probabilities) <= 0:
            self.total_reward += self.grid_world.get_reward(self.current)
            return self.current
        chosen_action = random.choices(actions, probabilities)[0]

        if self.end():
            return self.current

        actions = {
            'u':((max(self.current[0] - 1, 0), self.current[1]), [('l', (self.current[0], max(self.current[1] - 1, 0))), ('r', (self.current[0], min(self.current[1] + 1, self.grid_world.grid_size_y - 1))) ]),
            'd':((min(self.current[0] + 1, self.grid_world.grid_size_x - 1), self.current[1]), [('l', (self.current[0], max(self.current[1] - 1, 0))), ('r', (self.current[0], min(self.current[1] + 1, self.grid_world.grid_size_y - 1))) ]),
            'l':((self.current[0], max(self.current[1] - 1, 0)), [('u', (max(self.current[0] - 1, 0), self.current[1])), ('d', (min(self.current[0] + 1, self.grid_world.grid_size_x - 1), self.current[1])) ]),
            'r':((self.current[0], min(self.current[1] + 1, self.grid_world.grid_size_y - 1)), [('u', (max(self.current[0] - 1, 0), self.current[1])), ('d', (min(self.current[0] + 1, self.grid_world.grid_size_x - 1), self.current[1])) ])
        }
        
        new_state, sideways_states = actions[chosen_action]

        # Set default transitions for main actions to current state
        results = [(self.current, self.prob_forward)]
        
        if self.grid_world.get_type(new_state) != 'O':
            results = [(new_state, self.prob_forward)]
        
        # Add possible sideway transitions
        for sideway_action, sideway_state in sideways_states:
            if self.grid_world.get_type(sideway_state) != 'O':
                results.append((sideway_state, self.prob_sideways))
            else:
                results.append((self.current, self.prob_sideways))
        
        probabilities = []
        for each_result in results:
            probabilities.append(each_result[1])
        result = random.choices(results, probabilities)[0]
        
        self.current = result[0]
        self.total_reward += self.grid_world.get_reward(self.current)
        return self.current
    
    def transition_probabilities(self, state: tuple, action: str) -> dict[tuple, float]:
        """Given a state and an action, return the possible next states and their corresponding probabilities."""
        
        actions = {
            'u':((max(state[0] - 1, 0), state[1]), [('l', (state[0], max(state[1] - 1, 0))), ('r', (state[0], min(state[1] + 1, self.grid_world.grid_size_y - 1))) ]),
            'd':((min(state[0] + 1, self.grid_world.grid_size_x - 1), state[1]), [('l', (state[0], max(state[1] - 1, 0))), ('r', (state[0], min(state[1] + 1, self.grid_world.grid_size_y - 1))) ]),
            'l':((state[0], max(state[1] - 1, 0)), [('u', (max(state[0] - 1, 0), state[1])), ('d', (min(state[0] + 1, self.grid_world.grid_size_x - 1), state[1])) ]),
            'r':((state[0], min(state[1] + 1, self.grid_world.grid_size_y - 1)), [('u', (max(state[0] - 1, 0), state[1])), ('d', (min(state[0] + 1, self.grid_world.grid_size_x - 1), state[1])) ])
        }
        
        new_state, sideways_states = actions[action]
        
        # Set default transitions for main actions to current state
        results = {state: self.prob_forward,}
        
        if self.grid_world.get_type(new_state) != 'O':
            results = {new_state: self.prob_forward,}
        
        # Add possible sideway transitions
        for sideway_action, sideway_state in sideways_states:
            if self.grid_world.get_type(sideway_state) != 'O':
                results[sideway_state] = self.prob_sideways
            else:
                results[state] = self.prob_sideways
        
        return results

if __name__ == "__main__":
    from mdp import MDP

    layout_file = "map.txt"
    env = GridWorld(layout_file)
    rand_policy = PolicyGrid(env, (0.25, 0.25, 0.25, 0.25))
    mdp = MDP(env, discount=0.99, prob_forward=0.8, prob_sideways=0.1)
    mdp.value_iteration()
    mdp_policy = mdp.get_optimal_policy()
    
    sim = Simulator(env, 0.8, 0.1)
    sim.start()
    print(sim.current)
    for i in range(30):
        if sim.end():
            break
        sim.step(rand_policy)
        print(i, sim.current)
    print("Reward: ", sim.total_reward)

    sim.start()
    print(sim.current)
    for i in range(30):
        if sim.end():
            break
        sim.step(mdp_policy)
        print(i, sim.current)
    print("Reward: ", sim.total_reward)
