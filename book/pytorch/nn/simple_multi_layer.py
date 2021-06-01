import torch


def activation(x):
    """ Sigmoid activation function

        Arguments
        ---------
        x: torch.Tensor
    """
    return 1 / (1 + torch.exp(-x))


def simple_multi_layer_network():
    n_input = 3  # Number of input units, must match number of input features
    n_hidden = 2  # Number of hidden units
    n_output = 1  # Number of output units
    features = torch.randn((1, n_input))

    # Weights for inputs to hidden layer
    W1 = torch.randn(n_input, n_hidden)
    # Weights for hidden layer to output layer
    W2 = torch.randn(n_hidden, n_output)

    # and bias terms for hidden and output layers
    B1 = torch.randn((1, n_hidden))
    B2 = torch.randn((1, n_output))

    hidden0 = activation(torch.mm(features, W1) + B1)
    output = activation(torch.mm(hidden0, W2) + B2)
    print(output)


if __name__ == '__main__':
    print(torch.cuda.is_available())
    simple_multi_layer_network()
