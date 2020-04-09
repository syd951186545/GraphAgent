from torch import nn
import torch
import torch.nn.functional as F
from Configuration import config


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.num_neighbors = None
        self.FS = nn.Sequential(nn.Linear(3 * config.entity_dim + config.relation_dim, config.entity_dim),
                                nn.Tanh(),
                                # nn.Linear(2 * config.entity_dim, config.entity_dim),
                                # nn.ReLU()
                                )
        # self.FS.weight.data.normal_(0, 0.5)
        self.FP = nn.Linear(config.entity_dim, 1)

        # self.FP.weight.data.normal_(0, 0.5)
        self.FP2 = nn.Sequential(nn.Linear(2 * config.entity_dim, config.entity_dim),
                                 nn.ReLU(),
                                 nn.Linear(config.entity_dim, 1),
                                 )

        # self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def get_index(self, num_neighbors):
        self.num_neighbors = num_neighbors

    def forward(self, state_HS, hats):
        # state_HS = [batch,128+64+64]
        # u0,uks = [batch,]
        tempHS = self.FS(state_HS)
        u0 = self.FP(tempHS)
        index = 0
        Q_values = []
        for i in range(len(self.num_neighbors)):
            # 采用全连接网络
            # left = tempHS[i].expand(hats[index:index + self.num_neighbors[i], ].size())
            # catinput = torch.cat((left, hats[index:index + self.num_neighbors[i], ]), 1)
            # catinput = catinput.contiguous()
            # uk = self.FP2(catinput)
            # uk = uk.transpose(0, 1)
            # Q_values.append(self.sigmoid(torch.cat((u0[i].unsqueeze(0), uk), 1)))
            # 采用向量乘法
            uk = tempHS[[i]].mm(hats[index:index + self.num_neighbors[i], ].t())
            Q_values.append(self.sigmoid(torch.cat((u0[i].unsqueeze(0), uk), 1)))
        # 如果需要返回 action and prob
        # action_probilities = self.softmax(torch.cat((u0, uks), 1))
        # action_node = self.action_candidate[action_probilities.argmax()]
        # action_prob = torch.max(action_probilities)

        return Q_values
