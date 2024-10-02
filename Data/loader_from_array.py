import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from medmnist import PneumoniaMNIST
from torchvision import transforms


class MedMNIST_Testing(Dataset):
    """MedMNIST dataset class"""

    def __init__(self, input_arr):
        """Initialize MedMNISTDataset."""

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        labels = np.load('/media/zoe/ssd/pneumoniamnist_224/test_labels.npy')

        self.x, self.targets = input_arr, labels.squeeze()

    def __getitem__(self, index: int):
        """Return an item by the index."""
        img = self.x[index]

        img = Image.fromarray(img)
        img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        y = self.targets[index]

        return img, y

    def __len__(self) -> int:
        """Return the len of the dataset."""
        return len(self.x)