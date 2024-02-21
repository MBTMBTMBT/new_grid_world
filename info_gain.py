if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import tqdm
    from grid_world import GridWorld
    from mdp import MDP

    layout = "small.txt"
    prior_layout = "small_oneway.txt"

    # plt.figure()

    env = GridWorld(layout)
    mdp = MDP(env, discount=1, prob_forward=1)
    mdp.value_iteration()
    state_action_tans = env.get_state_action_transitions(1)
    start_state = env.find_positions('S')[0]

    prior_env = GridWorld(prior_layout)
    prior_mdp = MDP(prior_env, discount=1, prob_forward=1)
    prior_mdp.value_iteration()

    mdp_policy = mdp.get_optimal_policy()
    prior_policy = prior_mdp.get_optimal_policy()
    mdp_policy.visualize()
    prior_policy.visualize()

    info_mdp_grid = mdp.info_mdp_with_prior(state_action_tans, mdp_policy, prior_policy)
    value_grid = mdp.evaluate_policy(mdp_policy)
    info_start = info_mdp_grid[start_state[0], start_state[1]]
    value_start = value_grid[start_state[0], start_state[1]]
    
    mdp.visualize_info()

    plt.show()
