if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import tqdm
    from grid_world import GridWorld
    from mdp import MDP

    layout_files = ["small_maze___.txt", "small_mazex__.txt", "small_maze_x_.txt", "small_maze__x.txt"]  
    policy_strengths = np.arange(0.2, 1.0, 1e-2)

    plt.figure(figsize=(12, 6))

    # First subplot
    plt.subplot(1, 2, 1)

    # Second subplot
    plt.subplot(1, 2, 2)

    for layout_file in layout_files:
        env = GridWorld(layout_file)
        mdp = MDP(env, discount=1, prob_forward=1)
        mdp.value_iteration()

        # mdp.get_optimal_policy().visualize()

        state_action_tans = env.get_state_action_transitions(1)
        start_state = env.find_positions('S')[0]

        info_starts = []
        value_starts = []

        for ps in tqdm.tqdm(policy_strengths):
            mdp_policy = mdp.get_optimal_policy().interpolate(ps)
            info_mdp_grid = mdp.info_mdp(state_action_tans, mdp_policy)
            value_grid = mdp.evaluate_policy(mdp_policy)
            info_start = info_mdp_grid[start_state[0], start_state[1]]
            value_start = value_grid[start_state[0], start_state[1]]

            info_starts.append(info_start)
            value_starts.append(value_start)

        # Plotting for first subplot
        plt.subplot(1, 2, 1)
        plt.plot(info_starts, value_starts, '-o', label=f'File: {layout_file}')
        
        # Plotting for second subplot
        plt.subplot(1, 2, 2)
        gradient = np.gradient(np.array(value_starts), np.array(info_starts))
        plt.plot(info_starts, gradient, '-o', label=f'Gradient: {layout_file}')

    # First subplot settings
    plt.subplot(1, 2, 1)
    plt.xlabel('Info Start')
    plt.ylabel('Value Start')
    plt.title('Info vs Value Across Different Maps')
    plt.grid(True)
    plt.legend()

    # Second subplot settings
    plt.subplot(1, 2, 2)
    plt.xlabel('Info Start')
    plt.ylabel('Gradient of Value Start')
    plt.title('Gradient Across Different Maps')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # layout_file = "small_maze.txt"
    # env = GridWorld(layout_file)
    # mdp = MDP(env, discount=1, prob_forward=1)
    # mdp.value_iteration()
    # state_action_tans = env.get_state_action_transitions(1)
    # mdp_policy = mdp.get_optimal_policy(prob_forward=1)
    # mdp_policy.visualize()

    # info_mdp_grid = mdp.info_mdp(state_action_tans, mdp_policy)
    # value_grid = mdp.evaluate_policy(mdp_policy)

    # fig, ax = plt.subplots(figsize=(10, 10))
    # plt.title('Control Info')
    # ax.set_xticks(np.arange(mdp.info_mdp_grid.shape[1]))
    # ax.set_yticks(np.arange(mdp.info_mdp_grid.shape[0]))
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_xticks(np.arange(mdp.info_mdp_grid.shape[1]+1)-.5, minor=True)
    # ax.set_yticks(np.arange(mdp.info_mdp_grid.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    # ax.tick_params(which="minor", size=0)
    # directions = ['u', 'd', 'l', 'r']
    # for k in range(mdp.info_mdp_grid.shape[0]):
    #     for m in range(mdp.info_mdp_grid.shape[1]):
    #         value = mdp.info_mdp_grid[k, m].item()
    #         text = "%.2f" % value
    #         ax.text(m, k, text, ha='center', va='center', fontsize=10, color="black")
    # plt.gca().invert_yaxis()

    # fig, ax = plt.subplots(figsize=(10, 10))
    # plt.title('State Value')
    # ax.set_xticks(np.arange(mdp.value_grid.shape[1]))
    # ax.set_yticks(np.arange(mdp.value_grid.shape[0]))
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_xticks(np.arange(mdp.value_grid.shape[1]+1)-.5, minor=True)
    # ax.set_yticks(np.arange(mdp.value_grid.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    # ax.tick_params(which="minor", size=0)
    # directions = ['u', 'd', 'l', 'r']
    # for k in range(mdp.value_grid.shape[0]):
    #     for m in range(mdp.value_grid.shape[1]):
    #         value = mdp.value_grid[k, m].item()
    #         text = "%.2f" % value
    #         ax.text(m, k, text, ha='center', va='center', fontsize=10, color="black")
    # plt.gca().invert_yaxis()

    # plt.show()

    # policy_strenths = [i for i in range(0.5, 1.0, 2e-2)]
    # betas = [10**i for i in range(-5, 0.2, 1)]
    # start_state = env.find_positions('S')[0]
    # end_state = env.find_positions('T')[0]

    # for ps in policy_strenths:
    #     for beta in betas:
    #         mdp_policy = mdp.get_optimal_policy(prob_forward=ps)
    #         info_mdp_grid = mdp.info_mdp(state_action_tans, mdp_policy)
    #         value_grid = mdp.evaluate_policy(mdp_policy)
    #         info_start = info_mdp_grid[start_state[0], start_state[1]]
    #         value_start = value_grid[start_state[0], start_state[1]]
    #         free_energy = info_start - beta * value_start

    # policy_strenths = [i for i in range(0.5, 1.0, 2e-2)]
    # start_state = env.find_positions('S')[0]
    # end_state = env.find_positions('T')[0]

    # for ps in policy_strenths:
    #     mdp_policy = mdp.get_optimal_policy(prob_forward=ps)
    #     info_mdp_grid = mdp.info_mdp(state_action_tans, mdp_policy)
    #     value_grid = mdp.evaluate_policy(mdp_policy)
    #     info_start = info_mdp_grid[start_state[0], start_state[1]]
    #     value_start = value_grid[start_state[0], start_state[1]]

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D

    # # Your given data ranges
    # policy_strengths = np.arange(0.5, 1.0, 2e-2)
    # betas = np.array([10**i for i in range(-5, 1)])
    # start_state = env.find_positions('S')[0]
    # end_state = env.find_positions('T')[0]

    # # Create mesh grids
    # X, Y = np.meshgrid(policy_strengths, betas)

    # # Initialize Z (free_energy values) grid
    # Z = np.zeros_like(X)

    # # Calculate Z values
    # for i, ps in enumerate(policy_strengths):
    #     for j, beta in enumerate(betas):
    #         mdp_policy = mdp.get_optimal_policy(prob_forward=ps)
    #         info_mdp_grid = mdp.info_mdp(state_action_tans, mdp_policy)
    #         value_grid = mdp.evaluate_policy(mdp_policy)
    #         info_start = info_mdp_grid[start_state[0], start_state[1]]
    #         value_start = value_grid[start_state[0], start_state[1]]
    #         free_energy = info_start - beta * value_start
    #         Z[j, i] = free_energy

    # # Plotting
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, cmap='viridis')

    # ax.set_xlabel('Policy Strengths')
    # ax.set_ylabel('Betas')
    # ax.set_zlabel('Free Energy')

    # plt.show()
