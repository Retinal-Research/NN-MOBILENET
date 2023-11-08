from PIL import Image
import numpy as np
import pandas as pd
import os
from glob import glob
from torch.utils import data
from torch.utils.data import Dataset

class Messidor1Dataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data = pd.read_csv(label_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.iloc[idx, 1]
        label = self.data.iloc[idx, 2]
        if label > 1: 
            y = 1
        else:
            y = 0
        #y = label
        image_dir = os.path.join(self.image_dir, image_name)
        x = Image.open(image_dir)

        if self.transform:
            x = self.transform(x)

        label_onehot = np.zeros(2)
        label_onehot[y] = 1

        return x, label_onehot,y

    def get_labels(self):
        return self.data.iloc[:, 1]


class Messidor2Dataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data = pd.read_csv(label_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        if label > 1: 
            y = 1
        else:
            y = 0
        #y = label
        image_dir = os.path.join(self.image_dir, image_name)
        x = Image.open(image_dir)

        label_onehot = np.zeros(2)
        label_onehot[y] = 1

        if self.transform:
            x = self.transform(x)

        return x, label_onehot,y

    def get_labels(self):
        return self.data.iloc[:, 1]