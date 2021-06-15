import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def show_one_in_detail(image):
    img = np.squeeze(image)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max() / 2.5
    for x in range(width):
        for y in range(height):
            val = round(img[x][y], 2) if img[x][y] != 0 else 0
            ax.annotate(str(val), xy=(y, x), horizontalalignment='center', verticalalignment='center',
                        color='white' if img[x][y] < thresh else 'black')
    plt.show()


if __name__ == '__main__':
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 20

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()

    # choose the training and test datasets
    train_data = datasets.MNIST(root='~/junk/', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='~/junk/', train=False, download=True, transform=transform)

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images = images.numpy()

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap='gray')
        # print out the correct label for each image.item() gets the value contained in a Tensor
        ax.set_title(str(labels[idx].item()))

    plt.show()

    show_one_in_detail(images[1])
