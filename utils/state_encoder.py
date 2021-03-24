import torch.nn as nn


class StateEncoder(nn.Module):
    def __init__(self, sa_sizes, hidden_dim=128, norm_in=True):
        super(StateEncoder, self).__init__()
        self.nagents = len(sa_sizes)
        self.state_encoders = nn.ModuleList()
        for sdim, adim in sa_sizes:
            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                            hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

    def forward(self, states, agents=None, logger=None, niter=0):
        if agents is None:
            agents = range(self.nagents)
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]
        return [s_encodings[idx] for idx in agents]