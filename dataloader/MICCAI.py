from PIL import Image
import numpy as np
import pandas as pd
import os
from glob import glob
from torch.utils import data
from torch.utils.data import Dataset


class MICCAI(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        data = pd.read_csv(label_dir)
        self.image_all = data.iloc[:, 0].values
        self.label_all = data.iloc[:, 1].values
        print(data)
    def __getitem__(self, idx):
        #print(self.image_all[idx])
        image_name = str(self.image_all[idx]) #+ '.png'
        #print(self.data.iloc[idx, 0])
        label = self.label_all[idx]

        image_dir = os.path.join(self.image_dir, image_name)
        x = Image.open(image_dir)

        if self.transform:
            x = self.transform(x)

        label_onehot = np.zeros(5)
        label_onehot[label] = 1

        return x, label_onehot,label

    def get_labels(self):
        return self.label_all

    def __len__(self):
        return len(self.label_all)