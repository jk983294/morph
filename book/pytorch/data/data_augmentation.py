import torch
from torchvision import datasets, transforms
from book.pytorch.utils import helper
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_dir = '~/junk/Cat_Dog_data/'
    """
    randomly rotate, mirror, scale, and/or crop your images during training. 
    This will help your network generalize as it's seeing the same images but in different locations, 
    with different sizes, in different orientations, etc
    """
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5],
                                                                [0.5, 0.5, 0.5])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])

    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)  # dataloader is a generator
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    # Looping through it, get a batch on each loop
    # for images, labels in trainloader:
    #     pass

    # Get one batch
    images, labels = next(iter(trainloader))
    ax = helper.imshow(images[0], normalize=False)
    plt.show()

    data_iter = iter(testloader)
    images, labels = next(data_iter)
    fig, axes = plt.subplots(figsize=(10, 4), ncols=4)
    for ii in range(4):
        ax = axes[ii]
        helper.imshow(images[ii], ax=ax, normalize=False)
    plt.show()
