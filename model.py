import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self, network_params, device):
        super(Network, self).__init__()
        self.params = network_params
        self.device = device

        self.hidden = nn.Linear(self.params.state_dim,self.params.hidden_dim)
        self.output = nn.Linear(self.params.hidden_dim,self.params.action_dim)

    def forward(self, x):
        x = F.leaky_relu(self.hidden(x))
        out = self.output(x)
        return out
