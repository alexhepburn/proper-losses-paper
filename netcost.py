import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
from chromagram import *
from tqdm import tqdm
import pickle
import random

class R_norm(nn.Module):
    def __init__(self, r, cuda=False):
        super(R_norm, self).__init__()
        self.r = r
        self.cuda = cuda

    def forward(self, input, target, s, C_matrix, instance_weights):
        p = Variable(torch.zeros(target.shape[0], 13)).type(torch.cuda.FloatTensor)
        p.scatter_(1, target.view(-1, 1), 1)
        if self.cuda:
            p.cuda()
        z_pe=torch.bmm(C_matrix, p.unsqueeze(2)).squeeze()
        z_p = (s - z_pe)
        z_p_norm = torch.norm(z_p, self.r, 1)
        z_ye = (torch.bmm(C_matrix, input.unsqueeze(2)).squeeze())
        z_y = (s - z_ye)
        z_y_norm = torch.norm(z_y, self.r, 1)
        diag = torch.diag(torch.mm(z_y, z_p.transpose(0, 1)))
        return torch.sum((z_p_norm - 1/z_y_norm * diag))

class cross_entropy(nn.Module):
    def __init__(self, cuda=False):
        super(cross_entropy, self).__init__()
        self.cuda = cuda
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        y = Variable(torch.zeros(target.shape[0], 13)).cuda()
        y.scatter_(1, target.view(-1, 1), 1)
        t = - (y * self.logsoftmax(input))
        return torch.mean(torch.sum(t))

class instance_weight_ce(nn.Module):
    def __init__(self, cuda=False):
        super(instance_weight_ce, self).__init__()
        self.cuda = cuda
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, instance_weights):
        y = Variable(torch.zeros(target.shape[0], 13))
        if self.cuda:
            y.cuda()
        y.scatter_(1, target.view(-1, 1), 1)
        t = - (y * self.logsoftmax(input))
        return torch.sum(torch.sum(t, 1) * instance_weights)


class instance_class_weight_ce(nn.Module):
    def __init__(self, cuda=False):
        super(instance_class_weight_ce, self).__init__()
        self.cuda = cuda
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, class_weights, instance_weights):
        y = Variable(torch.zeros(target.shape[0], 13)).cuda()
        y.scatter_(1, target.view(-1, 1), 1)
        t = - (y * self.logsoftmax(input))
        c = torch.bmm(class_weights, t.unsqueeze(2)).squeeze()
        return torch.sum(torch.sum(c, 1)*instance_weights)

class class_weight_ce(nn.Module):
    def __init__(self, cuda=False):
        super(instance_class_weight_ce, self).__init__()
        self.cuda = cuda
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, class_weights):
        y = Variable(torch.zeros(target.shape[0], 13))
        if self.cuda:
            y.cuda()
        y.scatter_(1, target.view(-1, 1), 1)
        t = - (y * self.logsoftmax(input))
        c = class_weights * t
        return torch.sum(torch.sum(c, 1))

class Convnet(nn.Module):
    def __init__(self):
        super(Convnet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(8))
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 5, stride=(2, 1), padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16))
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 5, stride=(2, 1), padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32))
        self.fc1 = nn.Linear(32*250*12, 100)
        self.fc2 = nn.Linear(100, 13)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        prob = F.softmax(x, dim=1)
        return x, prob

class Mlp(nn.Module):
    def __init__(self, input_size):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 2)

    def forward(self, x, training=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        prob = F.softmax(x, dim=1)
        return x, prob
