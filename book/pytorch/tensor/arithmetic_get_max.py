import torch
import numpy as np

if __name__ == '__main__':
    x = torch.randn(3, 3, dtype=torch.float)
    y = torch.randn(3, 3, dtype=torch.float)
    print(x)
    print(y)

    max_idx = torch.argmax(x, 1)  # 按行
    print(max_idx)
    print(torch.gather(y, 1, max_idx.view(-1, 1)))
