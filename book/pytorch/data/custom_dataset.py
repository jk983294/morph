import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)  # two column csv: (pic_name.jpg,y_value)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """the number of samples in our dataset"""
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == '__main__':
    training_data = CustomImageDataset('~/junk/dummy/train/y.csv', '~/junk/dummy/train')
    test_data = CustomImageDataset('~/junk/dummy/test/y.csv', '~/junk/dummy/test')
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    train_X, train_y = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_X.size()}")
    print(f"Labels batch shape: {train_y.size()}")
