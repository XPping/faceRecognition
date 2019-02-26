import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

import torch
from torch.utils import model_zoo
from torch import nn
import re
from collections import OrderedDict
import torch.nn.functional as F

class DeepFace(nn.Module):

    def __init__(self):
        super(DeepFace, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=11, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=9, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=9, stride=1, padding=0, groups=16)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=7, stride=2, padding=0, groups=16)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=0, groups=16)

        self.fc1 = nn.Linear(16 * 21 * 21, 4096)
        self.fc2 = nn.Linear(4096, 4030)

        self.alpha = nn.Parameter(torch.Tensor(4030))
        self.alpha.data.normal_(0, 1.0)

    def forward(self, images1, images2):
        h = self.conv1(images1)
        h = self.pool1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = h.view(-1, h.size(1) * h.size(2) * h.size(3))
        h = self.fc1(h)
        f1 = self.fc2(h)
        h = self.conv1(images2)
        h = self.pool1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = h.view(-1, h.size(1) * h.size(2) * h.size(3))
        h = self.fc1(h)
        f2 = self.fc2(h)
        out = torch.abs(f1 - f2) * self.alpha
        out = torch.sum(out, dim=1)
        return out