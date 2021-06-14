import torch
import numpy as np

if __name__ == '__main__':
    x = torch.tensor([float('nan'), float('inf'), -float('inf'), 3.14])
    print(torch.nan_to_num(x))
