import torch

if __name__ == '__main__':
    x = torch.ones((2, 2), dtype=torch.float64)
    print(x.shape)
    print(x.size())
    print(torch.numel(x))  # total number of elements
    print(x.stride())
    print(x.dtype)
    print(x.is_floating_point)
    print(x.is_complex)
