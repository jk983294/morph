import torch
from book.pytorch.io import fc_model

if __name__ == '__main__':
    checkpoint = torch.load('/tmp/model.pth')
    model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    print(model)
