import torch
import numpy as np
from book.pytorch.utils.helper import get_mnist_loader
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt


class AutoEncoder(nn.Module):
    def __init__(self, encoding_dim):
        super(AutoEncoder, self).__init__()
        # encoder - linear layer (784 -> encoding_dim)
        self.fc1 = nn.Linear(28 * 28, encoding_dim)

        # decoder - linear layer (encoding_dim -> input size)
        self.fc2 = nn.Linear(encoding_dim, 28 * 28)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # (sigmoid for scaling from 0 to 1)
        return x


if __name__ == '__main__':
    """
    the compressed representation often holds key information about an input image and we can use it for 
    denoising images or oher kinds of reconstruction and transformation!
    """
    batch_size = 20
    train_loader, test_loader, valid_loader = get_mnist_loader(batch_size=batch_size, is_norm=False)

    model = AutoEncoder(encoding_dim=32)
    print(model)

    """comparing pixel values in input and output images, it's best to use a loss that meant for a regression task"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 20
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        for data in train_loader:
            images, _ = data
            images = images.view(images.size(0), -1)  # flatten images
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        # print avg training statistics
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    # check test
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    images_flatten = images.view(images.size(0), -1)
    output = model(images_flatten)
    images = images.numpy()

    # output is resized into a batch of images
    output = output.view(batch_size, 1, 28, 28)
    # use detach when it's an output that requires_grad
    output = output.detach().numpy()

    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25, 4))

    # input images on top row, reconstructions on bottom
    for images, row in zip([images, output], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()
