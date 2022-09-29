import argparse
import torch
from neurodata.load_data import create_dataloader
import numpy as np
import sys
from itertools import islice, chain, tee

from utils.train import train_epoch_bayesian
from models.DECOLLEModels import Network
from utils.test import test_bayesian, compute_ece
from utils.misc import make_experiment_dir
from utils.loss import DECOLLELoss
from optim.utils import get_optimizer
import optim.SampleGradEngine as SampleGradEngine

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--home', default=r"\users\home", type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_period', type=int, default=10)
    parser.add_argument('--num_ite', default=3, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--reg', default=0., type=float)
    parser.add_argument('--thr', default=1.25, type=float)
    parser.add_argument('--scale_grad', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--dt', type=int, default=50000)

    parser.add_argument('--num_samples_test', default=10, type=int)
    parser.add_argument('--rho', type=float, default=0.0000001)
    parser.add_argument('--burn_in', type=int, default=10)

    parser.add_argument('--fixed_prec', action='store_true', default=False)
    parser.add_argument('--initial_prec', default=1., type=float)
    parser.add_argument('--prior_m', type=float, default=0.)
    parser.add_argument('--prior_s', default=1., type=float)

    parser.add_argument('--binary', action='store_true', default=False)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--prior', default=0.5, type=float)

    parser.add_argument('--device', type=int, default=None)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        if args.device is not None:
            device = torch.device('cuda:%d' % args.device)
    else:
        device = torch.device('cpu')

    if args.fixed_prec:
        args.thr = 64 * args.thr
        weight_scale = 64
        scale = 1
    else:
        weight_scale = 1
        scale = 1 << 6

    if args.binary:
        synapses = 'binary'
    else:
        synapses = 'real_valued'

    results_path = make_experiment_dir(args.home + '/results',
                                       'mnistdvs_bayesian_decolle_nepochs_%d_'
                                       % args.num_epochs + synapses)

    with open(results_path + '/commandline_args.txt', 'w') as f:
        f.write('\n'.join(sys.argv[1:]))

    # Create dataloaders
    digits = [i for i in range(10)]

    dataset_path = args.home + r"/datasets/mnist-dvs/mnist_dvs_events_new.hdf5"
    for ite in range(args.num_ite):
        tasks_seen = []
        accs_mode = []
        accs_ens = []
        accs_comm = []
        ece_ens = []
        ece_comm = []

        acc_best = 0
        input_shape = 2 * 26 * 26
        net = Network(input_shape=input_shape,
                      hidden_shape=[1024, 512, 256],
                      output_shape=10,
                      scale_grad=args.scale_grad,
                      thr=args.thr,
                      burn_in=args.burn_in,
                      thr_scaling=args.binary).to(device)
        SampleGradEngine.add_hooks(net)

        optimizer = get_optimizer(net, args, device,
                                  binary_synapses=args.binary)
        loss_fn = DECOLLELoss(torch.nn.CrossEntropyLoss(), net, args.reg)

        train_dl, test_dl = create_dataloader(dataset_path,
                                              batch_size=args.batch_size,
                                              size=[input_shape],
                                              classes=digits,
                                              n_classes=10,
                                              dt=args.dt,
                                              shuffle_test=False,
                                              num_workers=0)

        for epoch in range(args.num_epochs):
            print('Epoch %d / %d' % (epoch + 1, args.num_epochs))

            train_iterator = iter(train_dl)
            loss = train_epoch_bayesian(net, loss_fn,
                                        optimizer, train_iterator,
                                        device, args.binary)

            if (epoch + 1) % args.test_period == 0:
                # Get mode/ensemble/committee test acc on the current task
                test_acc_mode, test_preds_mode, \
                test_acc_ens, test_preds_ens, \
                test_acc_comm, test_preds_comm, \
                true_labels_test \
                    = test_bayesian(net, test_dl,
                                    args.num_samples_test,
                                    optimizer, device)
                print('Mode acc at epoch %d for current task: %f'
                      % (epoch + 1, test_acc_mode))

                print('ensemble acc at epoch %d for current task: %f' % (
                    epoch + 1, test_acc_ens))

                print('committee acc at epoch %d for current task: %f' % (
                    epoch + 1, test_acc_comm))

                test_ece_ens = compute_ece(test_preds_ens, 20,
                                           true_labels_test
                                           )
                test_ece_comm = compute_ece(test_preds_comm, 20,
                                            true_labels_test
                                            )

                accs_mode.append(test_acc_mode)
                accs_ens.append(test_acc_ens)
                accs_comm.append(test_acc_comm)
                ece_ens.append(test_ece_ens)
                ece_comm.append(test_ece_comm)

                np.save(
                    results_path + '/test_preds_ensemble_ite_%d.npy'
                    % ite, test_preds_ens.detach().numpy())
                np.save(
                    results_path +'/test_preds_committee_ite_%d.npy'
                    % ite, test_preds_comm.detach().numpy())
                np.save(
                    results_path + '/true_labels_test_ite_%d.npy'
                    % ite, true_labels_test.detach().numpy())

                torch.save(optimizer.state_dict(),
                           results_path + '/optim_state_dict_ite_%d.pt')
