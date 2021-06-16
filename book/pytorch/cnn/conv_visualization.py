import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from book.pytorch.utils.helper import viz_layer


class Net(nn.Module):
    def __init__(self, weight):
        super(Net, self).__init__()
        # initializes the weights of the convolutional layer to be the weights of the 4 defined filters
        k_height, k_width = weight.shape[2:]
        # assumes there are 4 grayscale filters
        self.conv = nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)
        self.conv.weight = torch.nn.Parameter(weight)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)
        pooled_x = self.pool(activated_x)
        return conv_x, activated_x, pooled_x


def visualize_filters(filters):
    fig = plt.figure(figsize=(10, 5))
    for i in range(4):
        ax = fig.add_subplot(1, 4, i + 1, xticks=[], yticks=[])
        ax.imshow(filters[i], cmap='gray')
        ax.set_title('Filter %s' % str(i + 1))
        width, height = filters[i].shape
        for x in range(width):
            for y in range(height):
                ax.annotate(str(filters[i][x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center',
                            color='white' if filters[i][x][y] < 0 else 'black')
    plt.show()


if __name__ == '__main__':
    img_path = os.path.expanduser('~/junk/Cat_Dog_data/train/cat/cat.0.jpg')

    bgr_img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # normalize, rescale entries to lie in [0,1]
    gray_img = gray_img.astype("float32") / 255
    # plt.imshow(gray_img, cmap='gray')
    # plt.show()

    filter_vals = np.array([[-1, -1, 1, 1],
                            [-1, -1, 1, 1],
                            [-1, -1, 1, 1],
                            [-1, -1, 1, 1]])

    # define four filters
    filter_1 = filter_vals
    filter_2 = -filter_1
    filter_3 = filter_1.T
    filter_4 = -filter_3
    filters = np.array([filter_1, filter_2, filter_3, filter_4])

    visualize_filters(filters)

    weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
    model = Net(weight)
    print(model)

    gray_img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(1)
    conv_layer, activated_layer, pooled_layer = model(gray_img_tensor)
    viz_layer(conv_layer, n_filters=4)
    viz_layer(activated_layer, n_filters=4)
    viz_layer(pooled_layer, n_filters=4)
