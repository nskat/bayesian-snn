import lava.lib.dl.slayer as slayer
import numpy as np
import torch
from utils.misc import calculate_fan_in


class Network(torch.nn.Module):
    def __init__(self, input_shape, hidden_shape,
                 output_shape, burn_in=0,
                 scale_grad=3., tau_grad=0.03,
                 thr=1.25, bias=False,
                 scale=1 << 6, weight_scale=1,
                 thr_scaling=False):
        super(Network, self).__init__()

        self.burn_in = burn_in

        neuron_params = {
            'scale': scale,
            'threshold': thr,
            'current_decay': 0.25,
            'voltage_decay': 0.03,
            'tau_grad': tau_grad,
            'scale_grad': scale_grad,
            'requires_grad': False,
            'persistent_state': True
        }
        neuron_params_drop = {**neuron_params}

        weight_scale = weight_scale
        self.blocks = torch.nn.ModuleList()
        self.readout_layers = torch.nn.ModuleList()
        self.synapse_scales = []

        hidden_shape = [input_shape] + hidden_shape
        for i in range(len(hidden_shape)-1):
            self.blocks.append(slayer.block.cuba.Dense(
                neuron_params_drop, hidden_shape[i], hidden_shape[i+1],
                weight_norm=False, weight_scale=weight_scale))
            readout = torch.nn.Linear(hidden_shape[i+1],
                                      output_shape,
                                      bias=False)
            readout.weight.requires_grad = False

            self.readout_layers.append(readout)

        if thr_scaling:
            for block in self.blocks:
                threshold = thr * calculate_fan_in(block.synapse.weight)
                block.neuron._threshold \
                    = int(np.sqrt(threshold) * block.neuron.w_scale) \
                    / block.neuron.w_scale

    def forward(self, spike):
        spike.requires_grad_()
        spikes = []
        readouts = []
        voltages = []

        for block in self.blocks:
            z = block.synapse(spike.detach())
            _, voltage = block.neuron.dynamics(z)
            voltages.append(voltage)

            spike = block.neuron.spike(voltage)
            spikes.append(spike)

        for ro, spike in zip(self.readout_layers, spikes):
            readout = []
            for t in range(spike.shape[-1]):
                readout.append(ro(spike[..., t]))
            readouts.append(torch.stack(readout, dim=-1))

        return spikes, readouts, voltages

    def init_state(self, inputs, burn_in=None):
        self.reset_()
        if burn_in is None:
            burn_in = self.burn_in

        self.forward(inputs[..., :burn_in])
        return inputs[..., burn_in:]

    def reset_(self):
        for block in self.blocks:
            block.neuron.current_state[:] = 0.
            block.neuron.voltage_state[:] = 0.
