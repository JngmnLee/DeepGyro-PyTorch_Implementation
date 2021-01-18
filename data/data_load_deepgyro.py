from PIL import Image
import os
import os.path
from torchvision.transforms import functional as F
from torchvision import transforms
import torch.utils.data
from data import PairCompose
import numpy as np


def train_dataloader(path, batch_size, num_workers=0, use_transform=True):
    image_dir = os.path.join(path, 'train')
    transform = None
    if use_transform:
        transform = transforms.Compose(
             [transforms.ToTensor()]
        )
    dataloader = torch.utils.data.DataLoader(
        DeepGyroDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'valid')
    dataloader = torch.utils.data.DataLoader(
        DeepGyroDataset(image_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    dataloader = torch.utils.data.DataLoader(
        DeepGyroDataset(image_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return dataloader


class DeepGyroDataset(torch.utils.data.Dataset):

    def __init__(self, datapath, transform=None):
        self.root_dir = datapath
        self.image_list = os.listdir(os.path.join(datapath, 'blurred/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        blurred = Image.open(os.path.join(self.root_dir, 'blurred', self.image_list[idx]))
        blurx = Image.open(os.path.join(self.root_dir, 'blurx', self.image_list[idx]))
        blury = Image.open(os.path.join(self.root_dir, 'blury', self.image_list[idx]))
        label = Image.open(os.path.join(self.root_dir, 'sharp', self.image_list[idx]))

        if self.transform:
            blurred = self.transform(blurred)
            blurx = self.transform(blurx)
            blury = self.transform(blury)
            label = self.transform(label)
        else:
            blurred = F.to_tensor(blurred)
            blurx = F.to_tensor(blurx)
            blury = F.to_tensor(blury)
            label = F.to_tensor(label)

        image = torch.cat((blurred, blurx, blury), dim=0)

        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            _, ext = x.split('.')
            if ext not in ['png', 'jpg', 'jpeg']:
                raise ValueError
