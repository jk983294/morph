import torch
import numpy as np


def cuda_init():
    cuda0 = torch.device('cuda:0')
    print(torch.ones([2, 2], dtype=torch.float64, device=cuda0))


if __name__ == '__main__':
    print(torch.tensor([[1., -1.], [1., -1.]]))
    print(torch.tensor(np.array([[1, 2, 3], [4, 5, 6]])))
    print(torch.zeros([2, 2], dtype=torch.int32))
    print(torch.eye(2, 2, dtype=torch.int32))
    print(torch.arange(1., 4.))  # tensor([1., 2., 3.])
    print(torch.linspace(0, 1, steps=5))  # tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
    print(torch.diag(torch.randn(3)))
    print(torch.diagflat(torch.randn(2, 2)))
    print(torch.tensor([1, 2]).repeat(4, 2))
    print(torch.take(torch.tensor([[4, 3, 5], [6, 7, 8]]), torch.tensor([0, 2, 5])))
    # print(torch.tile(torch.tensor([1, 2]), (2, 2)))

    # cuda_init()

    """下面的new_*函数, type从原tensor继承,产生新的tensor"""
    tensor = torch.ones((2,), dtype=torch.float64)
    print(tensor.new_tensor([[0, 1], [2, 3]]))  # 数据从data来, 类型已经定义好
    print(tensor.new_full((3, 4), 3.141592))  # tensor不变,产生新的
    print(tensor.new_empty((2, 3)))  # 里面的数据没有初始化, 垃圾值
    print(tensor.new_ones((2, 3)))
    print(tensor.new_zeros((2, 3)))

    """fill"""
    tensor = torch.eye(3, 3, dtype=torch.float64)
    tensor.index_fill_(1, torch.tensor([0]), -1)
    print(tensor)
