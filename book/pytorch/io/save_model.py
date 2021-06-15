<<<<<<< HEAD
import torch
from torch import nn, optim
from book.pytorch.io import fc_model
from book.pytorch.utils.helper import get_mnist_loader

if __name__ == '__main__':
    model = fc_model.Network(784, 10, [512, 256, 128])
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainloader, testloader, _ = get_mnist_loader()

    fc_model.train(model, trainloader, testloader, criterion, optimizer, epochs=2)

    print("Our model: \n\n", model, '\n')
    print("The state dict keys: \n\n", model.state_dict().keys())

    checkpoint = {'input_size': 784,
                  'output_size': 10,
                  'hidden_layers': [each.out_features for each in model.hidden_layers],
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, '/tmp/model.pth')
||||||| 0111d6c
=======
import torch
from torch import nn, optim
from book.pytorch.io import fc_model
from book.pytorch.utils.helper import get_mnist_loader

if __name__ == '__main__':
    model = fc_model.Network(784, 10, [512, 256, 128])
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainloader, testloader = get_mnist_loader()

    fc_model.train(model, trainloader, testloader, criterion, optimizer, epochs=2)

    print("Our model: \n\n", model, '\n')
    print("The state dict keys: \n\n", model.state_dict().keys())

    checkpoint = {'input_size': 784,
                  'output_size': 10,
                  'hidden_layers': [each.out_features for each in model.hidden_layers],
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, '/tmp/model.pth')
>>>>>>> 9d06c5028639fdc99582041dfd5366124fa47f4f
