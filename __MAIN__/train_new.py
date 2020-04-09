from copy import copy
from collections import namedtuple

from Configuration import config
from Script.buildNetworkGraph import get_graph
from Environment.environment_train import Environment
from Environment.environment_predict import Environment as Environment2
from Agent.DQN import DQN

GRAPH = get_graph()
Transition = namedtuple('Transition', ['state_seq', 'reward'])

# MCTS-p-UCB
env = Environment(GRAPH)
env.init_TreeList(config.dataSet + "/train.txt")

env_dev = Environment2(GRAPH)
env_dev.init_TreeList(config.dataSet + "/dev.txt")


def train():
    # agentP = DQN(config.model_dir+"/act_net7.model")
    agentP = DQN(config.model_dir+"/act_net21.model")
    reward_train = 0
    total = 1

    for i_ep in range(config.num_episodes):
        # node_t 是经历一次蒙特卡洛树搜索后根据UCB选择的最好下一节点,交替采用
        root = env.reset()
        # env.render() # 打印
        # for j in range(8):
        state_seq, reward = env.step(root, PolicyChooseNet=agentP.choose_action,
                                     method=config.walk_method_predict)
        if state_seq[-1] == -1:
            state_seq = state_seq[:-1]
        transition = Transition(state_seq, reward)
        agentP.store_transition(transition)
        reward_train += reward
        total += 1

        if i_ep % 400 == 0:
            agentP.writer.add_scalar('Positive_reward/PCUB', reward_train / total,
                                     global_step=i_ep // 400)
            agentP.update()
            # 验证集
            if i_ep % 40000 == 0:
                reward_dev = 0
                for node in env_dev.TreeList:
                    state_seq, reward = env_dev.step(node, PolicyChooseNet=agentP.choose_action,
                                                     method=config.walk_method_predict)
                    reward_dev += reward
                print(reward_dev / len(env_dev.TreeList))
                agentP.writer.add_scalar('dev_reward/10update', reward_dev / len(env_dev.TreeList),
                                         global_step=(i_ep // 40000) - 1)


if __name__ == '__main__':
    train()
    # path = [4, 46, 2116, 86]
