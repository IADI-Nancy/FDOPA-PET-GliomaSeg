import torch
from torch import nn


class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
        inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
        applied to each entry like this:
        l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
        If weights are None, all w will be 1.
        """
        super(DeepSupervisionWrapper, self).__init__()
        assert any([x != 0 for x in weight_factors]), "At least one weight factor should be != 0.0"
        self.weight_factors = tuple(weight_factors)
        self.loss = loss
        self.compound_loss = {}

    def forward(self, *args):
        assert all([isinstance(i, (tuple, list)) for i in args]), \
            f"all args must be either tuple or list, got {[type(i) for i in args]}"
        # we could check for equal lengths here as well, but we really shouldn't overdo it with checks because
        # this code is executed a lot of times!

        if self.weight_factors is None:
            weights = (1, ) * len(args[0])
        else:
            weights = self.weight_factors

        # TZAR: it shouldn't be much slower than original implementation since it also uses a for loop but we gathered compound losses
        weighted_loss = 0
        if hasattr(self.loss, 'compound_loss'):
            for l_i in self.loss.compound_loss:
                self.compound_loss[l_i] = 0
                 
        for i, inputs in enumerate(zip(*args)):
            if weights[i] != 0.0:
                weighted_loss += weights[i] * self.loss(*inputs)
                if hasattr(self.loss, 'compound_loss'):
                    for l_i in self.loss.compound_loss:
                        self.compound_loss[l_i] += weights[i] * self.loss.compound_loss[l_i]

        return weighted_loss
