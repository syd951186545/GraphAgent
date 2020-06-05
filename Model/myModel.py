from torch import nn
import torch
from Configuration import config
from Model.action_encoder import Action_encoder
from Model.TCN_encoder import TCN_encoder
from Model.policy_net import PolicyNet


def _Max(matrix):
    """
    取矩阵每一列最大值得到一行向量
    :param matrix:
    :return:
    """

    return matrix[torch.argmax(matrix, dim=0), torch.LongTensor(range(matrix.shape[1]))].view(1, matrix.shape[1])


class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.num_neighbors, self.action_index = None, None
        self.action_encoder = Action_encoder()
        # (kernel,dropout,input_channel,layers_channel)
        # 最后一层的感受野大小由kernel和d决定:2kd-2d-k+2,d=1,2,4,8
        self.state_encoder = TCN_encoder(3, 0.1, config.entity_dim,
                                         [config.entity_dim, 2 * config.entity_dim, 2 * config.entity_dim, config.entity_dim])
        self.policy_net = PolicyNet()

    def get_index(self, num_neighbors, action_index):
        self.num_neighbors, self.action_index = num_neighbors, action_index

    def forward(self, NEs, nodes, query):
        hats = self.action_encoder(NEs)
        index = 0

        # padding,最长10个节点，全补满for batch
        # state = torch.zeros(((9 - len(nodes)) * 3 + 1, nodes.size()[1]))
        # no padding
        # state = torch.zeros((1, nodes.size()[1]))
        for i in range(len(self.num_neighbors) - 1):
            node = nodes[[i]]
            HAt = _Max(hats[index:index + self.num_neighbors[i], :])
            if -1 == self.action_index[i]:
                raise Exception("-1终止动作不在候选实体中，请单独处理")
            hat = hats[index:index + self.num_neighbors[i], :][[self.action_index[i]]]
            if 0 == i:
                HAts = HAt
                state_seq = torch.cat((node, HAt, hat), 0)
            else:
                HAts = torch.cat((HAts, HAt), 0)
                state_seq = torch.cat((state_seq, node, HAt, hat), 0)

            index += self.num_neighbors[i]

        # 需要保留HAts,包括最后一个节点
        HAt = _Max(hats[index:, :])
        if index == 0:  # 只有一个节点的初始状态
            HAts = HAt
            state_seq = nodes[[-1]]
        else:
            HAts = torch.cat((HAts, HAt), 0)
            state_seq = torch.cat((state_seq, nodes[[-1]]), 0)

        # channel first trans
        state_seq = state_seq.transpose(0, 1).contiguous()
        state_seq = state_seq.view_as(state_seq)[None, :, :]

        # TCN encoding
        # --------------------------------------------
        self.state_encoder.get_state_index(len(nodes))
        hs = self.state_encoder(state_seq)
        # channel back
        hs = hs.squeeze(0).transpose(1, 0).contiguous()

        # policy net
        # ---------------------------------------------
        # query = query.expand(len(nodes), query.size()[1])
        state_HS = torch.cat((query, hs, HAts), 1)
        self.policy_net.get_index(num_neighbors=self.num_neighbors)
        Qsa = self.policy_net(state_HS, hats)

        return Qsa


class myQPolicy(nn.Module):
    def __init__(self):
        super(myQPolicy, self).__init__()
        self.policy = PolicyNet()

    def get_index(self, num_neighbors):
        return self.policy.get_index(num_neighbors=num_neighbors)

    def forward(self, state_code, hats):
        Q_values = self.policy(state_code, hats)
        return Q_values
