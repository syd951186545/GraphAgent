from Configuration import config
from torch import nn
import torch.nn.functional as F


class Action_encoder(nn.Module):
    def __init__(self, ):
        super(Action_encoder, self).__init__()
        # 8+16 - 8 relu
        self.liner = nn.Linear(config.entity_dim + config.relation_dim, config.entity_dim)
        # self.liner.weight.data.normal_(0, 0.5)

    def forward(self, x):
        hats = self.liner(x)
        return F.relu(hats)


if __name__ == '__main__':
    pass
