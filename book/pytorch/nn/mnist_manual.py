import numpy as np
import torch
from book.pytorch.utils.helper import get_mnist_loader
from book.pytorch.utils.tensor_utils import activation, my_softmax


def multi_Layer_NW(inputUnits, hiddenUnits, outputUnits):
    torch.manual_seed(7)  # Set the random seed so things are predictable

    # Define the size of each layer in our network
    n_input = inputUnits  # Number of input units, must match number of input features
    n_hidden = hiddenUnits  # Number of hidden units
    n_output = outputUnits  # Number of output units

    # Weights for inputs to hidden layer
    W1 = torch.randn(n_input, n_hidden)
    # Weights for hidden layer to output layer
    W2 = torch.randn(n_hidden, n_output)

    # and bias terms for hidden and output layers
    B1 = torch.randn((1, n_hidden))
    B2 = torch.randn((1, n_output))
    return W1, W2, B1, B2


def calc_output(features, W1, W2, B1, B2):
    h = activation(torch.matmul(features, W1).add_(B1))
    output = activation(torch.matmul(h, W2).add_(B2))
    return output


if __name__ == '__main__':
    trainloader, testloader = get_mnist_loader()

    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(type(images))
    print(images.shape)
    print(labels.shape)
    # plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
    # plt.show()

    # Features are flattened batch input
    features = torch.flatten(images, start_dim=1)
    W1, W2, B1, B2 = multi_Layer_NW(features.shape[1], 256, 10)

    out = calc_output(features, W1, W2, B1, B2)  # output of your network, should have shape (64,10)
    print(out)

    # Here, out should be the output of the network in the previous excercise with shape (64,10)
    probabilities = my_softmax(out)

    print(probabilities.shape)  # shape should be (64, 10)
    print(probabilities.sum(dim=0))  # it should sum to 1
