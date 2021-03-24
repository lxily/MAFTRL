import torch.nn as nn
from itertools import chain
from utils.state_encoder import StateEncoder
from utils.attention_model import AttentionModel


class AttentionMedium(nn.Module):
    def __init__(self, sa_sizes, hidden_dim=128, attend_heads=1):
        super(AttentionMedium, self).__init__()
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads
        self.state_encoder = StateEncoder(sa_sizes=sa_sizes, hidden_dim=hidden_dim)

        self.attention_model = AttentionModel(sa_sizes=sa_sizes, hidden_dim=hidden_dim)

        self.shared_modules = [self.attention_model, self.state_encoder]

    def forward(self, obs, agents=None, logger=None, niter=0):
        if agents is None:
            agents = range(len(self.sa_sizes))
        s_encodings = self.state_encoder(obs)
        return s_encodings

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)