import numpy as np
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torch

class German_cred(Dataset):

    def __init__(self, x, y, class_weights, amount, c_matrix_e, bregman=False):
        self.x = x
        self.y = np.asarray(y)
        self.bregman = bregman
        self.C_matrix = class_weights
        self.C_matrix_e = c_matrix_e
        self.amount = amount
        p = []
        for i in range(0, x.shape[0]):
            b = []
            for j in range(0, class_weights[i].shape[1]):
                b.append(np.max(class_weights[i][:, j]))
            p.append(np.asarray(b))
        self.class_weights = p

    def __getitem__(self, index):
        x = torch.from_numpy(np.array(self.x[index]))
        y = self.y[index]
        if self.bregman:
            class_weights = self.class_weights[index]
        else:
            #class_weights = self.class_weights[y, :].squeeze()
            class_weights = [1, 0.2]
        c_mat = torch.from_numpy(np.array(self.C_matrix[index]))
        c_mat_e = torch.from_numpy(np.array(self.C_matrix_e[index]))
        c = torch.from_numpy(np.array(class_weights))
        return {'x': x, 'y':y, 'c':c.type(torch.FloatTensor),
            'c_mat':c_mat.type(torch.FloatTensor), 'am':self.amount[index],
            'c_mat_e':c_mat_e.type(torch.FloatTensor)}

    def __len__(self):
        return (self.y.shape[0])
