from optim.BBSNN import BayesBiSNN
from optim.GaussianBayesSNN import GaussianBayesOptimizer


def get_optimizer(network, args, device, binary_synapses=False, initial_params=None):
    if binary_synapses:
        optimizer = BayesBiSNN(network.parameters(),
                               initial_latent_params=initial_params,
                               prior=args.prior,
                               lr=args.lr,
                               rho=args.rho,
                               tau=args.tau,
                               device=device)
    else:
        optimizer \
            = GaussianBayesOptimizer(network.parameters(),
                                     initial_params_mean=initial_params,
                                     initial_params_prec=args.initial_prec,
                                     prior_m=args.prior_m,
                                     prior_s=args.prior_s,
                                     fixed_prec=False, weight_decay=0.,
                                     lr=args.lr, rho=args.rho, device=device)

    return optimizer
