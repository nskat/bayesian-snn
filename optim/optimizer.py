import torch
from torch.optim.optimizer import required
from copy import deepcopy
from itertools import chain
from collections import defaultdict, abc as container_abcs


class BayesOptimizer(torch.optim.Optimizer):
    """
    Abstract class to create optimizers with a collection of
    parameter groups corresponding to bayesian parameters
    e.g. mean and variance/precision in the Gaussian case. 
    The collection of groups is stored in a :class:`dict`, 
    and corresponding states in another one with matching keys.
    Bayesian param groups contain Bayesian parameters and corresponding priors
    """
    def __init__(self, params, defaults, device=required):
        super(BayesOptimizer, self).__init__(params, defaults)
        self.device = device

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain.from_iterable((g['params'] for g in saved_groups)),
                      chain.from_iterable((g['params'] for g in groups)))}

        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                if param.is_floating_point():
                    value = value.to(param.dtype)
                value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            other_keys = [k for k in group.keys() if k not in ('params',
                                                               'state')]
            for key in other_keys:
                new_group[key] = group[key]
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    def add_bayesian_param_group(self, name, prior_groups, init_values=None):
        r"""Adds variational parameters to
        the :class:`BayesOptimizer` s `param_groups`.
        Arguments:
            name (string): name of the group of parameters
            prior_groups (dict): priors
            init_values (float): If not None, to what values
            initialize the parameters
        """
        bayesian_param_groups = [group for group in self.param_groups]

        # check priors is consistent with params + refactor if needed
        if isinstance(prior_groups, torch.Tensor):
            raise TypeError("priors arguments given to the optim should be "
                            "a scalar or a list of scalars, tensors"
                            " or dicts, but got " +
                            torch.typename(prior_groups))

        if isinstance(prior_groups, list):
            if isinstance(prior_groups[0], dict):
                assert len(prior_groups) == len(bayesian_param_groups), \
                    "If passed as a list of dicts," \
                    " the length of the prior_groups" \
                    " arguments should match the number of groups"
                assert len(prior_groups[0].keys()) == 1, "only one prior " \
                                                         "should be passed " \
                                                         "per parameter group"
                assert prior_groups[0].keys()[0] == 'priors_' + name, \
                    "the priors group name does not match the desired " \
                    "variational parameter name"
                [bayes_group.update(prior_group) for (prior_group, bayes_group)
                 in zip(prior_groups, bayesian_param_groups)]
            elif isinstance(prior_groups[0], torch.Tensor):
                assert len(bayesian_param_groups) == 1, \
                    "priors were passed as a list of Tensors but there is" \
                    "more than one bayesian param groups, how to distribute" \
                    "the priors among groups cannot be inferred"
                assert len(bayesian_param_groups[0]) == len(prior_groups), \
                    "prior_groups were passed as a list of " \
                    "Tensors but its length does not match that of" \
                    "bayesian param groups"
                bayesian_param_groups[0].update({'priors_' + name: prior_groups}
                                                )
            else:
                [bayes_group.update({'priors_' + name:
                                         [torch.zeros_like(p) + prior_group
                                          for p in bayes_group['params']
                                          if p.requires_grad]})
                 for (bayes_group, prior_group) in zip(bayesian_param_groups,
                                                       prior_groups)]
        else:
            [bayes_group.update({'priors_' + name:
                                     [torch.zeros_like(p) + prior_groups
                                      for p in bayes_group['params']
                                      if p.requires_grad]})
             for bayes_group in bayesian_param_groups]

        # check init_value is either a scalar or a list of
        # tensors or dicts of correct size + refactor if needed
        if init_values is not None:
            if isinstance(init_values, torch.Tensor):
                raise TypeError("init_values argument given to the optim "
                                "should be a scalar or a list of tensors,"
                                "dicts or scalars but got " +
                                torch.typename(init_values))
            if isinstance(init_values, list):
                if isinstance(init_values[0], dict):
                    assert len(init_values) == len(bayesian_param_groups), \
                        "If passed as a list of dicts," \
                        " the length of the init_values" \
                        " arguments should match the number of groups"
                    assert len(init_values[0].keys()) == 1, \
                        "only one init_value should" \
                        " be passed per parameter group"
                    assert init_values[0].keys()[0] == name, \
                        "the init_values group names do" \
                        " not match the desired " \
                        "variational parameter name"
                    [bayes_group.update(init_value) for
                     (init_value, bayes_group) in zip(init_values,
                                                      bayesian_param_groups)]
                elif isinstance(init_values[0], torch.Tensor):
                    assert len(bayesian_param_groups) == 1, \
                        "init_values were passed as a list of " \
                        "Tensors but there is" \
                        "more than one bayesian param groups, " \
                        "how to distribute" \
                        "the values among groups cannot be inferred"
                    assert len(bayesian_param_groups[0]) == len(init_values), \
                        "init_values were passed as a list of " \
                        "Tensors but its length does not match that of" \
                        "bayesian param groups"
                    bayesian_param_groups[0].update({name: init_values})
                else:
                    [bayes_group.update({name: [torch.zeros_like(p) + init_value
                                                for p in bayes_group['params']
                                                if p.requires_grad]
                                         })
                     for (bayes_group, init_value) in zip(bayesian_param_groups,
                                                          init_values)]
            else:
                [bayes_group.update({name: [torch.zeros_like(p) + init_values
                                            for p in bayes_group['params']
                                            if p.requires_grad]})
                 for bayes_group in bayesian_param_groups]
        else:
            [bayes_group.update({name: [p.detach().clone()
                                        for p in bayes_group['params']
                                        if p.requires_grad]})
             for bayes_group in bayesian_param_groups]

        return
