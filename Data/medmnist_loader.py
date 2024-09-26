import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from medmnist import PneumoniaMNIST
from torchvision import transforms


class MedMNISTDataset(Dataset):
    """MedMNIST dataset class"""

    def __init__(self, root='/media/zoe/ssd/', mode='train', low_size=28):
        """Initialize MedMNISTDataset."""

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        self.resize = transforms.Resize((low_size, low_size), antialias=True)

        try:
            x = np.load(root + '/' + mode + '_images.npy')
            y = np.load(root + '/' + mode + '_labels.npy')

        except FileNotFoundError:

            img_dir = ('/').join((root.split('/'))[:-1])
            data = PneumoniaMNIST(split=mode, download=True, root=img_dir, size=224)
            x = data.imgs
            y = data.labels

        self.x, self.targets = x, y.squeeze()

    def __getitem__(self, index: int):
        """Return an item by the index."""
        img = self.x[index]

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        # Input img gets downsized to set size
        input_img = self.resize(img)

        # to get 'baseline:' imput_img.resize(size=(), resample=Image.BICUBIC)
        return img, input_img, index

    def __len__(self) -> int:
        """Return the len of the dataset."""
        return len(self.x)