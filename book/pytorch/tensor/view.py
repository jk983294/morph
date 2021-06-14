import torch
import numpy as np


def contiguous_example():
    base = torch.tensor([[0, 1], [2, 3]])
    print(base.is_contiguous())  # True

    t = base.transpose(0, 1)  # `t` is a view of `base`. No data movement happened here.
    print(t.is_contiguous())  # False

    c = t.contiguous()  # enforce copying data when `t` is not contiguous
    print(c.is_contiguous())  # True


if __name__ == '__main__':
    """allows a tensor to be a View of an existing tensor"""
    t = torch.rand(4, 4)
    b = t.view(2, 8)
    print(t.storage().data_ptr() == b.storage().data_ptr())  # True

    # Modifying view tensor changes base tensor as well
    b[0][0] = 3.14
    print(t)

    contiguous_example()

    print(torch.as_strided(t, (2, 2), (1, 2)))
    print(torch.diagonal(t, 0))
    print(torch.diagonal(t, 1))
    print(torch.diagonal(t, -1))

    x = torch.tensor([[1], [2], [3]])
    print(x.expand(3, 4))
    print(x.expand(-1, 4))  # -1 means not changing the size of that dimension

    """re-shape"""
    x = torch.randn(3, 2, 1)
    print(x.shape)
    print(torch.movedim(x, 1, 0).shape)
    print('unflatten', x.unflatten(1, (1, 2)).shape)  # torch.Size([3, 1, 2, 1]), 第2维变成 1*2
    print(t.view(16).shape)  # torch.Size([16])
    print(t.view(-1, 8).shape)  # torch.Size([2, 8]), -1表示该维度计算来
    print(torch.flatten(t))
    # print(torch.ravel(t))

    print(t)
    print(torch.narrow(t, 0, 0, 2))  # 按行,取前两行
    print(torch.narrow(t, 1, 0, 2))  # 按列,取前两列
    print(torch.select(t, 0, 1))  # 按行,取第两行
    print(torch.select(t, 1, 1))  # 按列,取第两列
    print(torch.index_select(t, 0, torch.tensor([0, 2])))  # 按行取
    print(torch.index_select(t, 1, torch.tensor([0, 2])))  # 按列取
    print('unbind')
    print(torch.unbind(t, dim=0))  # 按行拆成tuple
    print(torch.unbind(t, dim=1))  # 按列拆成tuple
    print(torch.split(t, 2, dim=0))  # 按行拆, 2行一组
    print(torch.chunk(t, 2, dim=0))  # 按行拆, 2行一组
    # print(torch.tensor_split(torch.arange(8), 3))

    """转置"""
    print(torch.t(t))  # 前两维转置
    print(t.T)  # 转置, x.permute(n-1, n-2, ..., 0)

    x = torch.randn(2, 3, 5)
    print(x.size())  # torch.Size([2, 3, 5])
    print(x.permute(2, 0, 1).size())  # torch.Size([5, 2, 3])

    """把维度是1的去除"""
    x = torch.zeros(2, 1, 2, 1, 2)
    print(x.size())  # torch.Size([2, 1, 2, 1, 2])
    print(torch.squeeze(x).size())  # torch.Size([2, 2, 2])

    vec = torch.arange(1., 9)
    print(vec.unfold(0, 2, 2))  # unfold(dimension, size, step)

    """增加一个维度"""
    print(vec.shape)  # torch.Size([8])
    print(torch.unsqueeze(vec, 0).shape)  # torch.Size([1, 8])
