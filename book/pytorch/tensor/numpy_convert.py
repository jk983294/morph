import torch
import numpy as np

if __name__ == '__main__':
    a = np.random.rand(4, 3)
    b = torch.from_numpy(a)  # a and b share memory
    another_a = b.numpy()  # another_a and b share memory

    b.mul_(2)  # in place multiply
    print(b)
    print(a)  # Numpy array matches new values from Tensor
    print(another_a)
