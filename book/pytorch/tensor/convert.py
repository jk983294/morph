import torch

if __name__ == '__main__':
    tensor = torch.randn(2, 2)
    print(tensor)  # initially dtype=float32, device=cpu
    print(tensor.tolist())

    """改变 dtype """
    print(tensor.to(torch.float64))  # dtype=torch.float64

    """改变 device """
    print(torch.cuda.current_device())  # 0
    cuda0 = torch.device('cuda:0')
    print(tensor.to(cuda0))  # device='cuda:0'

    """change both"""
    print(tensor.to(cuda0, dtype=torch.float64))

    """change type as other"""
    other = torch.randn((), dtype=torch.float64, device=cuda0)
    print(tensor.to(other, non_blocking=True))
