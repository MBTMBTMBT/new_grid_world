if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from grid_world import GridWorld
    from policy import PolicyGrid
    from sampler import *
    from mdp import MDP

    
    sample_iter = 20
    max_mdp_iter = 1000
    t = 100

    layout_file = "map.txt"
    env = GridWorld(layout_file)
    rand_policy = PolicyGrid(env, (0.25, 0.25, 0.25, 0.25))
    mdp = MDP(env, discount=0.99, prob_forward=1)
    mdp.value_iteration()
    mdp_policy = mdp.get_optimal_policy(prob_forward=1)
    # sampler = Sampler(env, prob_forward=1.0, prob_sideways=0.0)
    # mdp_policy.visualize()

    state_action_tans = env.get_state_action_transitions(1)
    print(state_action_tans)
    # for x in range(env.grid_size_x):
    #     for y in range(env.grid_size_y):
    #         state_action_tans[(x,y)] = {}
    #         for ia, action in enumerate(['u', 'd', 'l', 'r']):
    #             state_action_tans[(x,y)][action] = {}
    #             for x1 in range(env.grid_size_x):
    #                 for y1 in range(env.grid_size_y):
    #                     trans = sampler.simulator.transition_probabilities((x,y), action)
    #                     if (x1,y1) in trans.keys():
    #                         state_action_tans[(x,y)][action][(x1,y1)] = trans[(x1,y1)]
    #                     else:
    #                         state_action_tans[(x,y)][action][(x1,y1)] = 0
    
    # print(state_action_tans)
    
    # exit()

    # import tqdm
    start_state = env.find_positions('S')[0]
    start_state_free_energy = []
    start_state_control_info = []
    betas = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    betas = [1e-15]
    for beta in betas:
        policy_ = mdp.info_rl(state_action_tans, beta, max_iter=max_mdp_iter)
        start_state_free_energy.append(mdp.free_energy[beta][start_state[0],start_state[1]])
        start_state_control_info.append(mdp.control_info[beta][start_state[0],start_state[1]])
    policy_.visualize()

    # plt.figure(figsize=(10, 10)) 
    # plt.plot(betas, start_state_free_energy, marker='o') 
    # for x, y in zip(betas, start_state_free_energy):
    #     plt.text(x, y, f'({x},{y})')
    # plt.xlabel('Betas')
    # plt.ylabel('Free Energy')
    # plt.xscale('log')
    # plt.grid(True)

    # plt.figure(figsize=(10, 10)) 
    # plt.plot(betas, start_state_control_info, marker='o') 
    # for x, y in zip(betas, start_state_control_info):
    #     plt.text(x, y, f'({x},{y})')
    # plt.xlabel('Betas')
    # plt.ylabel('Control Info')
    # plt.xscale('log')
    # plt.grid(True)

    plt.figure(figsize=(10, 10))

    # Plotting the first dataset (Free Energy)
    plt.plot(betas, start_state_free_energy, marker='o', label='Free Energy')
    # Adding data labels for Free Energy
    for x, y in zip(betas, start_state_free_energy):
        plt.text(x, y, f'({x:.2f},{y:.2f})')

    # Plotting the second dataset (Control Info)
    plt.plot(betas, start_state_control_info, marker='o', label='Control Info')
    # Adding data labels for Control Info
    for x, y in zip(betas, start_state_control_info):
        plt.text(x, y, f'({x:.2f},{y:.2f})')

    # Common settings for the combined plot
    plt.xlabel('Betas')
    plt.ylabel('Values')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()  # Adding a legend to differentiate between the two datasets

    # plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xticks(np.arange(mdp.free_energy[beta].shape[1]))
    ax.set_yticks(np.arange(mdp.free_energy[beta].shape[0]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks(np.arange(mdp.free_energy[beta].shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(mdp.free_energy[beta].shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    directions = ['u', 'd', 'l', 'r']
    for k in range(mdp.free_energy[beta].shape[0]):
        for m in range(mdp.free_energy[beta].shape[1]):
            value = mdp.free_energy[beta][k, m].item()
            text = "%.2f" % value
            ax.text(m, k, text, ha='center', va='center', fontsize=10, color="black")
    plt.gca().invert_yaxis()

    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.set_xticks(np.arange(mdp.control_info[beta].shape[1]))
    # ax.set_yticks(np.arange(mdp.control_info[beta].shape[0]))
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_xticks(np.arange(mdp.control_info[beta].shape[1]+1)-.5, minor=True)
    # ax.set_yticks(np.arange(mdp.control_info[beta].shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    # ax.tick_params(which="minor", size=0)
    # directions = ['u', 'd', 'l', 'r']
    # for k in range(mdp.control_info[beta].shape[0]):
    #     for m in range(mdp.control_info[beta].shape[1]):
    #         value = mdp.control_info[beta][k, m].item()
    #         text = "%.2f" % value
    #         ax.text(m, k, text, ha='center', va='center', fontsize=10, color="black")
    # plt.gca().invert_yaxis()

    plt.show()

    pass

    # print(np.shape(mdp.infotogo[:,:,:]))
    # print(mdp.infotogo[:,:,:])


    # fig, ax = plt.subplots(figsize=(10, 10))

    # ax.set_xticks(np.arange(mdp.infotogo[:,:,:].shape[1]))
    # ax.set_yticks(np.arange(mdp.infotogo[:,:,:].shape[0]))
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_xticks(np.arange(mdp.infotogo[:,:,:].shape[1]+1)-.5, minor=True)
    # ax.set_yticks(np.arange(mdp.infotogo[:,:,:].shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    # ax.tick_params(which="minor", size=0)
    # directions = ['u', 'd', 'l', 'r']
    # for i in range(mdp.infotogo[:,:,:].shape[0]):
    #     for j in range(mdp.infotogo[:,:,:].shape[1]):
    #         value = mdp.infotogo[:,:,:][i, j]
    #         text = "\n".join(["{}: {:.2f}".format(directions[k], v) for k, v in enumerate(value)])
    #         ax.text(j, i, text, ha='center', va='center', fontsize=10, color="black")
    # plt.gca().invert_yaxis()

    # fig, ax = plt.subplots(figsize=(10, 10))

    # ax.set_xticks(np.arange(mdp.infotogo[:,:,:].shape[1]))
    # ax.set_yticks(np.arange(mdp.infotogo[:,:,:].shape[0]))
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_xticks(np.arange(mdp.infotogo[:,:,:].shape[1]+1)-.5, minor=True)
    # ax.set_yticks(np.arange(mdp.infotogo[:,:,:].shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    # ax.tick_params(which="minor", size=0)
    # directions = ['u', 'd', 'l', 'r']
    # for i in range(mdp.infotogo[:,:,:].shape[0]):
    #     for j in range(mdp.infotogo[:,:,:].shape[1]):
    #         value = mdp.delta[:,:,:][i, j]
    #         text = "\n".join(["{}: {:.2f}".format(directions[k], v) for k, v in enumerate(value)])
    #         ax.text(j, i, text, ha='center', va='center', fontsize=10, color="black")
    # plt.gca().invert_yaxis()

    # plt.show()
