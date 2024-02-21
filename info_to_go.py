if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from grid_world import GridWorld
    from policy import PolicyGrid
    from simulator import Simulator
    from sampler import *
    from mdp import MDP

    
    sample_iter = 20
    max_mdp_iter = 100
    t = 100

    layout_file = "map.txt"
    env = GridWorld(layout_file)
    rand_policy = PolicyGrid(env, (0.25, 0.25, 0.25, 0.25))
    mdp = MDP(env, discount=0.99, prob_forward=0.8, prob_sideways=0.1)
    mdp.value_iteration()
    mdp_policy = mdp.get_optimal_policy(forward=1.0, elsewhere=0.0)
    sampler = Sampler(env, prob_forward=1.0, prob_sideways=0.0)

    mdp_policy.visualize()

    prior_states_rand, _ = sampler.sample(rand_policy, sample_iter, t)
    # visualize_state_probability_at_time_t(env, prior_states_rand, 6)

    states_rand, _ = sampler.sample(mdp_policy, sample_iter, t)
    # visualize_state_probability_at_time_t(env, states_rand, 6)

    prior_states = {}
    for i in range(t+1):
        prior_states[i] = {}
        for x in range(env.grid_size_x):
            for y in range(env.grid_size_y):
                prior_states[i][(x,y)] = state_probability_at_time_t(t, (x,y), prior_states_rand)
                prior_states[i][(x,y)] = 0.02

    state_action_tans = {}
    for x in range(env.grid_size_x):
        for y in range(env.grid_size_y):
            state_action_tans[(x,y)] = {}
            for ia, action in enumerate(['u', 'd', 'l', 'r']):
                state_action_tans[(x,y)][action] = {}
                for x1 in range(env.grid_size_x):
                    for y1 in range(env.grid_size_y):
                        trans = sampler.simulator.transition_probabilities((x,y), action)
                        if (x1,y1) in trans.keys():
                            state_action_tans[(x,y)][action][(x1,y1)] = trans[(x1,y1)]
                        else:
                            state_action_tans[(x,y)][action][(x1,y1)] = 0
    
    state_action_prob = {}
    for x in range(env.grid_size_x):
        for y in range(env.grid_size_y):
            state_action_prob[(x,y)] = {}
            for ia, action in enumerate(['u', 'd', 'l', 'r']):
                state_action_prob[(x,y)][action] = mdp_policy.action_probabilities((x,y))[action]

    mdp.info_to_go_iteration(prior_states, state_action_tans, state_action_prob, max_iter=max_mdp_iter, discount=0.8)
    # print(np.shape(mdp.infotogo[:,:,:]))
    # print(mdp.infotogo[:,:,:])


    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_xticks(np.arange(mdp.infotogo[:,:,:].shape[1]))
    ax.set_yticks(np.arange(mdp.infotogo[:,:,:].shape[0]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks(np.arange(mdp.infotogo[:,:,:].shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(mdp.infotogo[:,:,:].shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    directions = ['u', 'd', 'l', 'r']
    for i in range(mdp.infotogo[:,:,:].shape[0]):
        for j in range(mdp.infotogo[:,:,:].shape[1]):
            value = mdp.infotogo[:,:,:][i, j]
            text = "\n".join(["{}: {:.2f}".format(directions[k], v) for k, v in enumerate(value)])
            ax.text(j, i, text, ha='center', va='center', fontsize=10, color="black")
    plt.gca().invert_yaxis()

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_xticks(np.arange(mdp.infotogo[:,:,:].shape[1]))
    ax.set_yticks(np.arange(mdp.infotogo[:,:,:].shape[0]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks(np.arange(mdp.infotogo[:,:,:].shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(mdp.infotogo[:,:,:].shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    directions = ['u', 'd', 'l', 'r']
    for i in range(mdp.infotogo[:,:,:].shape[0]):
        for j in range(mdp.infotogo[:,:,:].shape[1]):
            value = mdp.delta[:,:,:][i, j]
            text = "\n".join(["{}: {:.2f}".format(directions[k], v) for k, v in enumerate(value)])
            ax.text(j, i, text, ha='center', va='center', fontsize=10, color="black")
    plt.gca().invert_yaxis()

    plt.show()
