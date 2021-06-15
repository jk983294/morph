import torch
from torchvision.transforms import Lambda

if __name__ == '__main__':
    """one-hot encoded tensor for y"""
    target_transform = Lambda(lambda y:
                              torch.zeros(0, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
