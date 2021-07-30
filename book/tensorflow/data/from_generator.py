import tensorflow as tf
from book.tensorflow.keras_demo.train.training import get_compiled_model
from skimage.io import imread
from skimage.transform import resize
import numpy as np


class CIFAR10Sequence(tf.keras.utils.Sequence):
    def __init__(self, file_names, labels, batch_size):
        self.file_names, self.labels = file_names, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.file_names) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.file_names[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([resize(imread(filename), (200, 200)) for filename in batch_x]), np.array(batch_y)


if __name__ == '__main__':
    model = get_compiled_model()

    file_names = []
    labels = []
    sequence = CIFAR10Sequence(file_names, labels, batch_size=64)
    model.fit(sequence, epochs=10)
