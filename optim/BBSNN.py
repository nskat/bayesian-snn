import torch
from torch.optim.optimizer import required

from .optimizer import BayesOptimizer


def binarize(x, magnitude=1.):
    x.data[x.data >= 0] = magnitude
    x.data[x.data < 0] = - magnitude


class BayesBiSNN(BayesOptimizer):
    def __init__(self, binary_params, initial_latent_params, prior, lr=required,
                 tau=required, rho=required, device=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, tau=tau, rho=rho)
        super(BayesBiSNN, self).__init__(binary_params, defaults)
        self.add_bayesian_param_group('latent', prior, initial_latent_params)
        self.device = device

    @torch.no_grad()
    def step(self):
        for i, group in enumerate(self.param_groups):
            lr = group['lr']
            rho = group['rho']
            tau = group['tau']

            params = [w_b for w_b in group['params'] if w_b.requires_grad]
            for w_b, w_r, prior in zip(params,
                                       group['latent'],
                                       group['priors_latent']):

                if w_b.grad is not None:
                    mu_squared = torch.tanh(w_r).pow(2)
                    scale = (1 - w_b.data.pow(2)) / tau / torch.max(1 - mu_squared + 1e-12)
                    w_r.mul_(1 - lr * rho).add_(w_b.grad * scale - rho * prior, alpha=-lr)

    @torch.no_grad()
    def update_weights(self, howto='train'):
        """"
        Make parameters in 'params' group binary following the distribution
        defined by the latent parameters
        howto (string) should be one of:
        `train`: weights are approximately binary using the
        Gumbel-Softmax trick
        `mode`: weights are computed using the most probable state
        `rand`: weights are randomly generated
        """
        assert howto in ['train', 'mode', 'rand'], 'mode should be one of' \
                                                 '`train`, `MAP` or `rand`'

        for group in self.param_groups:
            tau = group['tau']

            params = [w_b for w_b in group['params'] if w_b.requires_grad]
            for w_b, w_r in zip(params, group['latent']):
                if howto == 'train':
                    epsilon = torch.rand(w_r.data.shape, device=w_r.device)
                    delta = torch.log(epsilon / (1 - epsilon)) / 2
                    w_b.data = torch.tanh((delta + w_r.data) / tau)
                elif howto == 'mode':
                    w_b.data = 2 * torch.sigmoid(2 * w_r.data) - 1
                    binarize(w_b)
                elif howto == 'rand':
                    w_b.data = \
                        2 * torch.bernoulli(torch.sigmoid(2 * w_r.data)) - 1
    
    def update_priors(self):
        for group in self.param_groups:
            for latent_param, prior in zip(group['latent'], group['priors_latent']):
                prior.data = latent_param.detach().clone()
