import torch
import torch.nn as nn

class StateActionEncoder(nn.Module):
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True):
        super(StateActionEncoder, self).__init__()
        self.nagents = len(sa_sizes)
        self.critic_encoders = nn.ModuleList()
        for sdim, adim in sa_sizes:
            idim = sdim + adim
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim,affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)

    def forward(self, inps, agents=None, logger=None, niter=0):
        if agents is None:
            agents = range(self.nagents)
        inps = [torch.cat((s, a), dim=1) for s, a in inps]
        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        return [sa_encodings[idx] for idx in agents]

    def scale_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.critic_encoders.parameters():
            p.grad.data.mul_(1. / self.nagents)