import os
import numpy as np
import torch
from torch.utils.data import Dataset
# from skimage import io
from PIL import Image
import torchvision
# import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root_dir) for f in filenames if 'png' in f]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_name = self.image_files[index]
        label = int(img_name.split('/')[-2])
        image = Image.open(img_name)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return label, image

# train_dataset = CustomDataset(root_dir='data/etl_952_singlechar_size_64/952_train')
# test_dataset = CustomDataset(root_dir='data/etl_952_singlechar_size_64/952_test')

# if __name__ == "__main__":
#     print(train_dataset[2])
#     print(len(train_dataset))
#     label, image = train_dataset[2]
#     print(f"Label: {label}, Image shape: {image.shape}")
#     print(len(image))
