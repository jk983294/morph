import torch
import numpy as np

if __name__ == '__main__':
    M = torch.randn(3, 5)
    batch1 = torch.randn(10, 3, 4)
    batch2 = torch.randn(10, 4, 5)
    print(torch.addbmm(M, batch1, batch2))  # beta * M + alpha * batch1 * batch2

    M_batch = torch.randn(10, 3, 5)
    print(torch.baddbmm(M_batch, batch1, batch2).shape)  # torch.Size([10, 3, 5])
    print(torch.bmm(batch1, batch2).shape)  # batch1 * batch2
