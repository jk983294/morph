import torch
from torchvision import datasets, transforms

from book.pytorch.utils import helper
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_dir = '~/junk/Cat_Dog_data/train'
    # combine these transforms into a pipeline
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)  # dataloader is a generator

    # Looping through it, get a batch on each loop
    # for images, labels in dataloader:
    #     pass

    # Get one batch
    images, labels = next(iter(dataloader))
    ax = helper.imshow(images[0], normalize=False)
    plt.show()
