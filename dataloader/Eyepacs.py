from PIL import Image
import numpy as np
import pandas as pd
import os
from glob import glob
from torch.utils import data
from torch.utils.data import Dataset


class Eyepacs(Dataset):
    def __init__(self, image_dir, file_dir, split, val_test, transform=None):
        dataset = pd.read_csv(file_dir)
        labels = dataset.iloc[:, 1].values
        image_ids = dataset.iloc[:, 0].values

        if split == 'train':
            self.labels = labels
            self.image_dir = image_dir
            self.image_ids = image_ids
            self.transform = transform
        else:
            val_test_label = dataset.iloc[:, 2].values
            val_test_idx = np.where(val_test_label == val_test)[0]   
            self.labels = labels[val_test_idx]
            self.image_ids = image_ids[val_test_idx]
            self.image_dir = image_dir
            self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_name = self.image_ids[idx] + '.png'
        image = Image.open(os.path.join(self.image_dir,image_name))
        label = self.labels[idx]
        
        label_onehot = np.zeros(5)
        label_onehot[label] = 1
        
        if self.transform:
            image = self.transform(image)

        return image, label_onehot,label

    def get_labels(self):
        return self.labels

