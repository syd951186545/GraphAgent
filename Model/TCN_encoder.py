import torch.nn.functional as F
from torch import nn
from Model.tcn import TemporalConvNet


class TCN_encoder(nn.Module):
    def __init__(self, kernel_size, dropout, input_channel, num_channels=None):
        super(TCN_encoder, self).__init__()
        # num_channels= [],channel in each layers,layers_num = len(num_channels)
        if num_channels is None:
            num_channels = [8, 16, 16, 8]
        self.tcn = TemporalConvNet(input_channel, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.state_index = [-1]

    def get_state_index(self, node_num):
        self.state_index = [3 * i for i in range(node_num)]

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        states_hs = y1[:, :, self.state_index]
        return states_hs
