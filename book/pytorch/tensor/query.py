<<<<<<< HEAD
import torch

if __name__ == '__main__':
    x = torch.ones((2, 2), dtype=torch.float64)
    print(x.shape)
    print(x.size())
    print(torch.numel(x))  # total number of elements
    print(x.stride())
    print(x.dtype)
    print(f"Device tensor is stored on: {x.device}")
    print(x.is_floating_point)
    print(x.is_complex)
||||||| 0111d6c
=======
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
>>>>>>> 9d06c5028639fdc99582041dfd5366124fa47f4f
