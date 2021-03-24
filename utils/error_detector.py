import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from itertools import chain

MSELoss = torch.nn.MSELoss()


class ErrorDetector(nn.Module):
    def __init__(self, sa_sizes, realtime_tolerance=None, hidden_dim=None, norm_in=True, lr=0.01):
        super(ErrorDetector, self).__init__()
        self.nagents = len(sa_sizes)
        self.error_detectors = nn.ModuleList()
        assert realtime_tolerance is not None
        self.realtime_tolerance = realtime_tolerance
        for sdim, adim in sa_sizes:
            detector = nn.Sequential()
            if hidden_dim is None:
                hidden_dim = (sdim + 1) // 2
            if norm_in:
                detector.add_module('det_bn', nn.BatchNorm1d(sdim, affine=False))
            detector.add_module('det_fc1', nn.Linear(sdim, hidden_dim))
            detector.add_module('det_nl', nn.LeakyReLU())
            detector.add_module('det_fc2', nn.Linear(hidden_dim, sdim))
            # detector.add_module('det_nl2', nn.LeakyReLU())
            self.error_detectors.append(detector)

    def forward(self, states, agents=None, logger=None, niter=0):
        if agents is None:
            agents = range(self.nagents)
        # extract state encoding for each agent that we're returning Q for
        s_output = [self.error_detectors[a_i](states[a_i]) for a_i in agents]
        return [s_output[idx] for idx in agents]
