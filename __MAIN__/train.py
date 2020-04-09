from copy import copy
from collections import namedtuple

from Configuration import config
from Script.buildNetworkGraph import get_graph
from Environment.environment import Environment
from Environment.environment_predict import Environment as Environment2
from Agent.DQN import DQN

GRAPH = get_graph()
Transition = namedtuple('Transition', ['state_seq', 'reward'])
# MCTS-UCB
env1 = Environment(GRAPH)
env1.init_TreeList(config.dataSet + "/train.txt")
# MCTS-p-UCB
env2 = Environment2(GRAPH)
env2.init_TreeList(config.dataSet + "/train.txt")

env_dev = Environment2(GRAPH)
env_dev.init_TreeList(config.dataSet + "/dev.txt")


def train():
    # agentP = DQN(config.model_dir+"/act_net7.model")
    agentP = DQN(None)
    reward_train = reward_x = 0
    total = totalx = 1

    for i_ep in range(config.num_episodes):
        # node_t 是经历一次蒙特卡洛树搜索后根据UCB选择的最好下一节点,交替采用
        if (reward_train + reward_x) / (total + totalx) < 0.5:
            root = env1.reset()
            state_seq, reward = env1.step(root, PolicyChooseNet=agentP.choose_action,
                                          method="MCTS")
            transition = Transition(state_seq, reward)
            agentP.store_transition(transition)
            reward_train += reward
            total += 1
            # print(state_seq, reward)
        else:
            # state_seq, reward = env1.step(root, PolicyChooseNet=agentP.choose_action,
            # method="DQN_self")
            # print(state_seq, reward)
            root = copy(env2.reset())
            state_seq, reward = env2.step(root, PolicyChooseNet=agentP.choose_action,
                                          method=config.walk_method_predict)
            if state_seq[-1] == -1:
                state_seq = state_seq[:-1]
            transition = Transition(state_seq, reward)
            # print(state_seq, reward)
            agentP.store_transition(transition)
            reward_x += reward
            totalx += 1
            agentP.writer.add_scalar('Positive_reward/agent_self', reward_x / totalx,
                                     global_step=totalx)

        if i_ep % 400 == 0:
            agentP.writer.add_scalar('Positive_reward/agent_self+MCTS', (reward_train + reward_x) / (total + totalx),
                                     global_step=i_ep // 400)
            agentP.update()
            # 验证集
            if i_ep % 40000 == 0:
                reward_dev = 0
                for node in env_dev.TreeList:
                    state_seq, reward = env2.step(node, PolicyChooseNet=agentP.choose_action,
                                                  method="DQN_self")
                    reward_dev += reward
                print(reward_dev / len(env_dev.TreeList))
                agentP.writer.add_scalar('dev_reward/10update', reward_dev / len(env_dev.TreeList),
                                         global_step=(i_ep // 40000)-1)


if __name__ == '__main__':
    train()
    # path = [4, 46, 2116, 86]
