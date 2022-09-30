import torch
import tqdm
import numpy as np


@torch.no_grad()
def test_frequentist(net, dl, device):
    test_loader = iter(dl)
    predictions = torch.Tensor()
    true_labels = torch.FloatTensor()

    for inputs, label in tqdm.tqdm(test_loader):  # test
        net.eval()
        net.reset_()

        inputs = inputs.to(device)
        true_labels = torch.cat((true_labels, label))

        inputs = net.init_state(inputs)
        spikes, readouts, _ = net(inputs)
        preds = readouts[-1].cpu()

        preds = torch.mean(torch.softmax(preds, dim=-2), dim=(-1))
        predictions = torch.cat((predictions, preds))
        torch.cuda.empty_cache()

    acc = torch.mean((predictions.argmax(-1) == true_labels).type(torch.float))
    print('Spike rates: ', [torch.mean(spike) for spike in spikes])
    return acc, predictions, true_labels


def test_bayesian(net, test_dl, num_samples, optimizer, device):
    acc_mode, predictions_mode, true_labels = test_mode(net, optimizer,
                                                        test_dl, device)
    acc_ens, predictions_ens, _ = test_ensemble(net, test_dl, num_samples,
                                                optimizer, device)
    acc_comm, predictions_comm, _ = test_committee(net, test_dl,
                                                   num_samples, optimizer,
                                                   device)
    return acc_mode, predictions_mode, acc_ens, predictions_ens, \
           acc_comm, predictions_comm, true_labels


@torch.no_grad()
def test_mode(net, optimizer, test_dl, device):
    net.eval()
    optimizer.update_weights(howto='mode')
    predictions = torch.Tensor()
    test_loader = iter(test_dl)
    true_labels = torch.FloatTensor()

    for inputs, label in tqdm.tqdm(test_loader):
        net.reset_()
        inputs = inputs.to(device)
        true_labels = torch.cat((true_labels, label))

        inputs = net.init_state(inputs)
        spikes, readouts, _ = net(inputs)
        preds = readouts[-1].cpu()

        preds = torch.mean(torch.softmax(preds, dim=-2), dim=(-1))
        predictions = torch.cat((predictions, preds))
        torch.cuda.empty_cache()

    acc = torch.mean((predictions.argmax(-1) == true_labels).type(torch.float))
    print('Spike rates: ', [torch.mean(spike) for spike in spikes])
    return acc, predictions, true_labels


@torch.no_grad()
def test_ensemble(net, test_dl, num_samples, optimizer, device):
    net.eval()
    predictions = torch.FloatTensor()
    test_loader = iter(test_dl)
    true_labels = torch.FloatTensor()

    for inputs, label in tqdm.tqdm(test_loader):
        predictions_sample = torch.FloatTensor()

        inputs = inputs.to(device)
        true_labels = torch.cat((true_labels, label))
        for _ in range(num_samples):
            optimizer.update_weights(howto='rand')
            net.reset_()

            inputs_sample = net.init_state(inputs)
            spikes, readouts, _ = net(inputs_sample)
            preds = readouts[-1].cpu()

            preds = torch.mean(torch.softmax(preds, dim=-2), dim=(-1))
            predictions_sample = torch.cat((predictions_sample,
                                            preds.unsqueeze(0)))
            torch.cuda.empty_cache()

        predictions = torch.cat((predictions,
                                 predictions_sample), dim=1)

    acc = torch.mean(
        (torch.mean(predictions, dim=0).argmax(-1) == true_labels).type(
            torch.float))
    print('Spike rates: ', [torch.mean(spike) for spike in spikes])
    return acc, predictions, true_labels


@torch.no_grad()
def test_committee(net, test_dl, num_samples, optimizer, device):
    net.eval()
    predictions = torch.FloatTensor()
    for _ in tqdm.tqdm(range(num_samples)):
        test_loader = iter(test_dl)
        predictions_sample = torch.FloatTensor()
        true_labels = torch.FloatTensor()

        optimizer.update_weights(howto='rand')

        for inputs, label in test_loader:
            net.reset_()
            inputs = inputs.to(device)
            true_labels = torch.cat((true_labels, label))

            inputs = net.init_state(inputs)
            spikes, readouts, _ = net(inputs)
            preds = readouts[-1].cpu()

            preds = torch.mean(torch.softmax(preds, dim=-2), dim=(-1))
            predictions_sample = torch.cat((predictions_sample, preds))
            torch.cuda.empty_cache()

        predictions = torch.cat((predictions,
                                 predictions_sample.unsqueeze(0)))

    acc = torch.mean(
        (torch.mean(predictions, dim=0).argmax(-1) == true_labels).type(
            torch.float))
    print('Spike rates: ', [torch.mean(spike) for spike in spikes])
    return acc, predictions, true_labels


@torch.no_grad()
def compute_ece(predictions, M, true_labels):
    predictions = torch.mean(predictions, dim=0)

    probas = predictions.max(-1).values
    predictions = predictions.argmax(-1)

    bins = np.arange(0, 1 + 1 / M, 1 / M)

    examples_per_bins = [[] for _ in range(M)]
    for i, proba in enumerate(probas):
        for m in range(M):
            if ((proba >= bins[m]) and (proba < bins[m + 1])):
                examples_per_bins[m].append(i)

    examples_in_bins = [example for bin_ in examples_per_bins for example in
                        bin_]

    assert len(examples_in_bins) == len(predictions), \
        "Number of examples in bins (%d) doesn't match the numbers of examples (%d)" % (
            len(examples_in_bins), len(predictions))

    acc_per_bin = [
        torch.mean((predictions[examples_per_bins[m]] == true_labels[
            examples_per_bins[m]]).type(torch.float)) for m in
        range(M)]
    acc_per_bin = [acc if not acc.isnan() else torch.tensor(0.) for acc in
                   acc_per_bin]
    conf_per_bin = [torch.mean(probas[examples_per_bins[m]]) for m in range(M)]
    conf_per_bin = [conf if not conf.isnan() else torch.tensor(0.) for conf in
                    conf_per_bin]

    ECE = torch.sum(
        torch.stack([len(examples_per_bins[m]) * torch.abs(
            acc_per_bin[m] - conf_per_bin[m]) for m in range(M)])) / len(
        predictions)
    return ECE.numpy()
