from torch import nn
from book.pytorch.utils.helper import get_mnist_loader


def logit_lose():
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    # Build a feed-forward network
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size))
    # Define the loss
    criterion = nn.CrossEntropyLoss()

    trainloader, testloader, _ = get_mnist_loader()
    images, labels = next(iter(trainloader))
    images = images.view(images.shape[0], -1)  # Flatten images

    # Forward pass, get our logits
    logits = model(images)
    # Calculate the loss with the logits and the labels
    loss = criterion(logits, labels)

    print(loss)


def log_softmax_lose():
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    # Build a feed-forward network
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size),
                          nn.LogSoftmax(dim=1))
    # Define the loss
    criterion = nn.NLLLoss()

    trainloader, testloader, _ = get_mnist_loader()
    images, labels = next(iter(trainloader))
    images = images.view(images.shape[0], -1)  # Flatten images
    print(images.shape)

    # Forward pass, get our log-probabilities
    logps = model(images)
    # Calculate the loss with the logps and the labels
    loss = criterion(logps, labels)

    print(loss)

    print('Before backward pass: \n', model[0].weight.grad)
    loss.backward()
    print('After backward pass: \n', model[0].weight.grad)


if __name__ == '__main__':
    logit_lose()
    log_softmax_lose()
