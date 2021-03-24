import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from utils.state_encoder import StateEncoder
from utils.attention_model import AttentionModel
from utils.state_action_encoder import StateActionEncoder


class AttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """

    def __init__(self, sa_sizes, hidden_dim=128, norm_in=True, attend_heads=4):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(AttentionCritic, self).__init__()
        # assert attend_heads == 1
        assert hidden_dim % attend_heads == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.state_encoder = StateEncoder(sa_sizes=sa_sizes, hidden_dim=hidden_dim)
        self.attention_model = AttentionModel(sa_sizes=sa_sizes, hidden_dim=hidden_dim)
        self.state_action_encoder = StateActionEncoder(sa_sizes=sa_sizes, hidden_dim=hidden_dim)

        self.critics = nn.ModuleList()
        # iterate over agents
        for sdim, adim in sa_sizes:
            odim = adim
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)

        self.shared_modules = [self.attention_model, self.state_action_encoder]

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

    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        if agents is None:
            agents = range(len(self.sa_sizes))
        states = [s for s, a in inps]
        actions = [a for s, a in inps]

        # extract state encoding for each agent that we're returning Q for
        s_encodings = self.state_encoder(states)
        # extract state-action encoding for each agent
        sa_encodings = self.state_action_encoder(inps)

        other_all_values, all_attend_logits, all_attend_probs = self.attention_model(s_encodings, sa_encodings,
                                                                                     sa_encodings)

        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze(1).sum(1)
                               .mean()) for probs in all_attend_probs[i]]
            agent_rets = []
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)

            all_q = self.critics[a_i](critic_in)

            int_acs = actions[a_i].max(dim=1, keepdim=True)[1]

            q = all_q.gather(1, int_acs)

            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)
            if regularize:
                # regularize magnitude of attention logits
                attend_mag_reg = 1e-3 * sum((logit ** 2).mean() for logit in
                                            all_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_rets.append(regs)
            if return_attend:
                agent_rets.append(np.array(all_attend_probs[i]))
            if logger is not None:
                logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                   niter)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets
