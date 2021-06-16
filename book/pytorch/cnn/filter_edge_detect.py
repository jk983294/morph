import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

if __name__ == '__main__':
    image = mpimg.imread(os.path.expanduser('~/junk/Cat_Dog_data/train/cat/cat.0.jpg'))
    plt.imshow(image)
    plt.show()

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    plt.imshow(gray, cmap='gray')
    plt.show()

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    filtered_y_image = cv2.filter2D(gray, -1, sobel_y)
    plt.imshow(filtered_y_image, cmap='gray')
    plt.show()

    filtered_x_image = cv2.filter2D(gray, -1, sobel_x)
    plt.imshow(filtered_x_image, cmap='gray')
    plt.show()
