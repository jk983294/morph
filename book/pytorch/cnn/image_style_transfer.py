import os
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from torchvision import transforms, models
from PIL import Image


def im_convert(tensor):
    """ un-normalizing an image, converting it from a Tensor image to a NumPy image for display"""
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


def load_image(img_path, max_size=400, shape=None):
    """ making sure the image is <= 400 pixels in the x-y dims."""
    image = Image.open(os.path.expanduser(img_path)).convert('RGB')

    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image


def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for a set of layers"""
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)  # 走过所有层
        if name in layers:  # 只记录感兴趣的层的结果
            features[layers[name]] = x
    return features


def show_images(a, b):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(im_convert(a))
    ax2.imshow(im_convert(b))
    plt.show()


def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor https://en.wikipedia.org/wiki/Gramian_matrix """
    b, d, h, w = tensor.size()  # batch_size, depth, height, and width
    tensor = tensor.view(b * d, h * w)
    gram = torch.mm(tensor, tensor.t())  # calculate the gram matrix
    return gram


def save_image(tensor, path_):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)  # H x W x C
    image = (image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))) * 255
    image = image.astype(np.uint8)
    image = image.clip(0, 255)

    # Tensor of shape C x H x W or a numpy ndarray of shape H x W x C to a PIL
    unloader_ = transforms.ToPILImage()
    image = unloader_(image)
    path_ = os.path.expanduser(path_)
    image.save(path_)
    print("save to", path_)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get the "features" portion of VGG19 (we will not need the "classifier" portion)
    vgg = models.vgg19(pretrained=True).features
    # freeze all VGG parameters since we're only optimizing the target image
    for param in vgg.parameters():
        param.requires_grad_(False)

    vgg.to(device)

    """Resize style to match content, makes code easier"""
    content = load_image('~/github/morph/resource/images/octopus.jpg').to(device)
    style = load_image('~/github/morph/resource/images/hockney.jpg', shape=content.shape[-2:]).to(device)

    # show_images(content, style)

    # print(vgg)  # 查看 network 获得需要的层定义
    layers = {'0': 'conv1_1',
              '5': 'conv2_1',
              '10': 'conv3_1',
              '19': 'conv4_1',
              '21': 'conv4_2',  # content representation
              '28': 'conv5_1'}

    # get content and style features only once before training
    content_features = get_features(content, vgg, layers)
    style_features = get_features(style, vgg, layers)

    # calculate the gram matrices for each layer of style representation
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # start off with the target as a copy of our *content* image then iteratively change its style
    target = content.clone().requires_grad_(True).to(device)

    # weights for each style layer
    # weighting earlier layers more will result in larger style artifacts
    # notice we are excluding 'conv4_2' our content representation
    style_weights = {'conv1_1': 1.,
                     'conv2_1': 0.75,
                     'conv3_1': 0.2,
                     'conv4_1': 0.2,
                     'conv5_1': 0.2}

    content_weight = 1  # alpha
    style_weight = 1e6  # beta

    optimizer = optim.Adam([target], lr=0.003)
    steps = 2000  # decide how many iterations to update your image
    show_every = 400
    for ii in range(1, steps + 1):
        target_features = get_features(target, vgg, layers)

        # the content loss = the mean squared difference between the target and content features at layer conv4_2
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

        # the style loss
        style_loss = 0
        for layer in style_weights:  # add up each layer's gram matrix loss
            # get the "target" style representation for the layer
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            style_gram = style_grams[layer]  # get the "style" style representation
            # the style loss for one layer, weighted appropriately
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_style_loss / (d * h * w)

        total_loss = content_weight * content_loss + style_weight * style_loss

        # update your target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # display intermediate images and print the loss
        print('round=%d, Total loss=%f' % (ii, total_loss.item()))
        if ii % show_every == 0:
            plt.imshow(im_convert(target))
            plt.show()

    show_images(content, target)
    save_image(content, "~/result.jpg")
