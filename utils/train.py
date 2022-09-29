import tqdm
import optim.SampleGradEngine as SampleGradEngine
import torch


def train_epoch_bayesian(network, loss_fn, optimizer, train_iter, device,
                         binary, reduction='mean'):
    network.train()
    loss = 0
    for (inputs, label) in tqdm.tqdm(train_iter):  # training loop
        network.reset_()
        inputs = inputs.to(device)
        label = label.to(device)

        inputs = network.init_state(inputs)
        for t in range(inputs.shape[-1]):
            optimizer.update_weights(howto='train')
            spikes, readouts, voltages = network(inputs[..., t].unsqueeze(-1))
            loss = loss_fn(readouts, voltages, label)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            del loss
            del spikes
            del readouts
            del voltages

        del inputs
        torch.cuda.empty_cache()

    return


def train_epoch_frequentist(network, loss_fn, optimizer, train_iter, device):
    network.train()
    loss = 0
    for (inputs, label) in tqdm.tqdm(train_iter):
        network.reset_()
        inputs = inputs.to(device)
        label = label.to(device)

        inputs = network.init_state(inputs)
        for t in range(inputs.shape[-1]):
            spikes, readouts, voltages = network(inputs[..., t].unsqueeze(-1))
            loss = loss_fn(readouts, voltages, label)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

    return loss
