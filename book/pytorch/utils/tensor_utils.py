import torch


def activation(x):
    """ Sigmoid activation function

        Arguments
        ---------
        x: torch.Tensor
    """
    return 1 / (1 + torch.exp(-x))


def my_softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=0)
