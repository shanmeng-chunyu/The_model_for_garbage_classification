from PIL import Image
import warnings
from PIL import Image, TiffImagePlugin
import torch
import torch.utils.data as data
import os
import random

warnings.filterwarnings("ignore", category=UserWarning, module=TiffImagePlugin.__name__)

class Dataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = sorted(os.listdir(self.root))
        class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.images = []
        for class_name in self.classes:
            class_idx = class_to_idx[class_name]
            class_path = os.path.join(self.root, class_name)
            if not os.path.isdir(class_path):
                continue
            for img in os.listdir(class_path):
                img_path = os.path.join(class_path, img)
                self.images.append((img_path, class_idx))

    def __getitem__(self, index):
        img_path, label = self.images[index]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img, label
        except OSError as e:
            new_index = random.randint(0, len(self) - 1)
            return self.__getitem__(new_index)

    def __len__(self):
        return len(self.images)