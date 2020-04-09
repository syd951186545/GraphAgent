import os
import sys
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

from Configuration import config
from Model.myModel import myModel
from Script.state_tup2input import transInput
from Script.Tools import _give_index

# Hyper-parameters
Transition = namedtuple('Transition', ['state_seq', 'reward'])
relation_dim = config.relation_dim


class DQN:
    capacity = config.capacity
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    gamma = config.gamma
    decay = config.decay  # learning_rate在200epoch后下降至0.13*1e-3
    memory_count = 0
    update_batch = 0

    def __init__(self, pre_model):
        super(DQN, self).__init__()
        self.target_net, self.act_net = myModel(), myModel()
        if pre_model:
            self.act_net.load_state_dict(torch.load(pre_model))

        self.memory = [None] * self.capacity
        self.optimizer = optim.Adam(self.act_net.parameters(), self.learning_rate)
        self.lr_decay = lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decay)
        self.loss_func = nn.MSELoss()
        self.writer = SummaryWriter(config.Summary_dir)

    def choose_action(self, state_tup):
        with torch.no_grad():
            NEs, nodes, query, num_neighbors, action_index, action_nodes = transInput(state_tup)

            self.act_net.get_index(num_neighbors, action_index)
            # 只取最后一位状态的结果，,选择q值最大的动作
            Q_values = self.act_net(NEs, nodes, query)[-1]
            # action = action_nodes[Q_values.max(1)[1]]
        return action_nodes[-1], Q_values

    def store_transition(self, transition):
        # !!transition = Transition(path, reward)
        index = self.memory_count % self.capacity
        self.memory[index] = transition
        self.memory_count += 1
        return self.memory_count >= self.capacity

    def update(self):
        print(self.memory_count)
        if self.memory_count < self.capacity:
            print("wait enough train date")
            return
        else:
            self.lr_decay.step()
            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), batch_size=self.batch_size,
                                      drop_last=False):
                losses = torch.tensor(0).float()
                self.optimizer.zero_grad()
                # batch update by one-one
                for i, j in enumerate(index):
                    # state = self.target_net.analysis_state(mem.state)
                    mem = self.memory[j]
                    state_tup = mem.state_seq
                    reward = torch.tensor([mem.reward]).float()

                    NEs, nodes, query, num_neighbors, action_index, action_nodes = transInput(state_tup)
                    # encoder and policy need to know actions
                    self.act_net.get_index(num_neighbors, action_index)
                    Q_values1 = self.act_net(NEs, nodes, query)
                    select_index = [index + 1 for index in action_index]
                    Q_value1 = _give_index(Q_values1, select_index)
                    with torch.no_grad():
                        # Double DQN
                        self.target_net.get_index(num_neighbors, action_index)
                        # next_state without the first state
                        Q_values2 = self.target_net(NEs, nodes, query)
                        if len(Q_values2) > 1:
                            Q_values2 = Q_values2[1:]
                            select_index = []
                            for i, qs in enumerate(Q_values1):
                                if i != 0:
                                    select_index.append(qs.max(1)[1])

                            Q_value2 = _give_index(Q_values2, select_index)
                            target_v = 0*reward + self.gamma * Q_value2
                            target_v = torch.cat((target_v, reward.unsqueeze(0)), 0)
                        else:
                            target_v = reward

                        # Nature DQN
                        # self.target_net.get_index(num_neighbors, action_index)
                        # # next_state without the first state
                        # Q_values2 = self.target_net(NEs, nodes, query)
                        # Q_values2

                    loss = self.loss_func(Q_value1, target_v)
                    loss.backward()
                    losses += loss.data / len(index)
                self.optimizer.step()
                self.writer.add_scalar('LOSS/batch', losses, self.update_batch)
                self.update_batch += 1
                if self.update_batch % config.net_replace == 0:
                    self.target_net.load_state_dict(self.act_net.state_dict())
                    if not os.path.exists(config.model_dir): os.mkdir(config.model_dir)
                    torch.save(self.act_net.state_dict(),
                               config.model_dir + "act_net" + str(self.update_batch//config.net_replace) + ".model")
