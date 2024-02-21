import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tqdm

from grid_world import GridWorld
from policy import PolicyGrid

class ProbMDP:
    def __init__(self, grid_world: GridWorld, policy_grid: PolicyGrid, prob_forward=0.8):
        self.grid_world = grid_world
        self.policy_grid = policy_grid
        self.prob_forward = prob_forward
        # self.threshold = threshold
        self.probability_distribution = np.zeros((self.grid_world.grid_size_x, self.grid_world.grid_size_y))
        self.state_action_transitions = grid_world.get_state_action_transitions(prob_forward)

    def calculate_probability_distribution(self, threshold=0.001, max_iter=1000):
        # 初始化概率分布，在起始点S的概率为1
        start_positions = self.grid_world.find_positions('S')
        for pos in start_positions:
            self.probability_distribution[pos] = 1.0 / len(start_positions)
        
        for iteration in tqdm.tqdm(range(max_iter)):
            delta = 0
            new_probability_distribution = np.zeros((self.grid_world.grid_size_x, self.grid_world.grid_size_y))
            
            # 在迭代中更新概率分布
            for x in range(self.grid_world.grid_size_x):
                for y in range(self.grid_world.grid_size_y):
                    state = (x, y)
                    current_prob = self.probability_distribution[state]
                    action_probs = self.policy_grid.action_probabilities(state)
                    
                    for action, action_prob in action_probs.items():
                        transitions = self.state_action_transitions[state][action]
                        for next_state, trans_prob in transitions.items():
                            # 累加转移到next_state的概率
                            new_probability_distribution[next_state] += current_prob * action_prob * trans_prob

            # 计算最大变化量以判断是否继续迭代
            for x in range(self.grid_world.grid_size_x):
                for y in range(self.grid_world.grid_size_y):
                    state = (x, y)
                    delta = max(delta, np.abs(new_probability_distribution[state] - self.probability_distribution[state]))
            
            print(new_probability_distribution)

            # 如果变化量小于阈值则停止迭代
            if delta < threshold:
                print(f"Converged after {iteration} iterations.")
                break
            
            # 更新概率分布
            self.probability_distribution = new_probability_distribution.copy()

    def get_probability_distribution(self):
        return self.probability_distribution

    def get_state_probability(self, state):
        return self.probability_distribution[state]
    
    def visualize_probabilities(self):
        fig, ax = plt.subplots()

        # 定义每个单元格类型的颜色
        cell_colors = {
            '.': 'grey',
            'T': 'yellow',
            'S': 'blue',
            'O': 'black',
            '+': 'green',
            'X': 'red'
        }

        # 创建颜色映射表
        cmap = mcolors.ListedColormap(list(cell_colors.values()))

        # 转换布局为数值类型以便着色
        data = np.zeros((self.grid_world.grid_size_x, self.grid_world.grid_size_y), dtype=int)
        for i in range(self.grid_world.grid_size_x):
            for j in range(self.grid_world.grid_size_y):
                cell_type = self.grid_world.layout[i, j]
                data[i, j] = list(cell_colors.keys()).index(cell_type)

        # 绘制网格世界
        ax.imshow(data, cmap=cmap)

        # 在每个单元格上显示概率值
        for i in range(self.grid_world.grid_size_x):
            for j in range(self.grid_world.grid_size_y):
                prob = self.probability_distribution[i, j]
                # 根据背景色决定文本颜色
                cell_type = self.grid_world.layout[i, j]
                text_color = 'white' if cell_colors[cell_type] in ['blue', 'black', 'red'] else 'black'
                ax.text(j, i, f'{prob:.4f}', ha='center', va='center', color=text_color)

        # plt.show()


if __name__ == "__main__":
    from mdp import MDP
    max_mdp_iter = 1000
    threshold=0.00000001

    layout_file = "map.txt"
    env = GridWorld(layout_file)
    rand_policy = PolicyGrid(env, (0.25, 0.25, 0.25, 0.25))

    mdp = MDP(env, discount=0.99, prob_forward=0.8)
    mdp.value_iteration()
    mdp_policy = mdp.get_optimal_policy(prob_forward=0.8)

    # 创建ProbMDP对象
    prob_mdp = ProbMDP(env, mdp_policy, prob_forward=0.8)  # 假设向前移动成功的概率是0.8

    # 计算概率分布
    prob_mdp.calculate_probability_distribution(threshold=threshold, max_iter=max_mdp_iter)

    # 可视化概率分布
    prob_mdp.visualize_probabilities()

    plt.show()
