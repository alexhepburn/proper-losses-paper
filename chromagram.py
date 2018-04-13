import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torch

class Chromagram(Dataset):

    def __init__(self, chroma, labels, instance_weights, class_weights=None,
         transform=None, bregman=False):
        self.transform = transform
        self.chroma = list(chroma)
        self.dim = self.chroma[0].shape
        self.labels = np.asarray(labels)
        self.instance_weights = np.asarray(instance_weights)
        self.bregman = bregman
        self.class_matrix = class_weights
        if bregman:
            b = []
            for j in range(0, class_weights.shape[1]):
                b.append(np.max(class_weights[:, j]))
            self.class_weights = np.asarray(b)
        else:
            self.class_weights = class_weights

    def __getitem__(self, index):
        c = np.expand_dims(np.squeeze(self.chroma[index]), axis=0)
        l = self.labels[index]
        C = self.class_matrix
        c_e = self.class_matrix * self.instance_weights[index]
        if self.instance_weights is None:
            instance_weights = 0
        else:
            instance_weights = self.instance_weights[index]
        if self.class_weights is None:
            class_weights = 0
        elif self.bregman:
            class_weights = self.class_weights
            class_vector = self.class_matrix[l, :]
        else:
            class_weights = self.class_weights[l, :]
            class_vector = 0
        if self.transform:
            c = self.transform(c)
        return {'x':torch.from_numpy(c), 'y': l,
            'w':instance_weights, 'c':class_weights, 'c_vec':class_vector, 'c_mat':C, 'c_mat_e':c_e}

    def __len__(self):
        return (len(self.chroma))
