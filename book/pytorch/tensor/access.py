import torch
import numpy as np

if __name__ == '__main__':
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(x[1][2])
    x[0][1] = 8
    print(x)

    # get a Python number from a tensor containing a single value
    print(x[1][2].item())
    print(torch.tensor([[1]]).item())
    print(x.tolist())
