from __future__ import print_function, division

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import cv2


class CnnEncoder(nn.Module):
    """
    Image encoder using a pretrained resnet18 model.
    Adds a resizing step to obtain encoded images of low dimensionaltiy, useful for computing similarity
    """

    def __init__(self, img_size=4):
        super(CnnEncoder, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True)

        # Preserve encoding layers only
        modules = list(resnet.children())[:-2]
        self.model = nn.Sequential(*modules)
                
        # Resize image to given size
        self.resize = nn.AdaptiveAvgPool2d((img_size, img_size))

    def forward(self, images):
        x = self.model(images)
        x = self.resize(x)
        x = x.mean(axis=1)
        return x