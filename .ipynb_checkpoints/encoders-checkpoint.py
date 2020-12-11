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
from PIL import Image

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
        
        self.mask = generate_mask()
        
        
    def forward(self, images):
        x = torch.unsqueeze(prepare_for_model(images, self.mask), 0)
        x = self.model(x)
        x = self.resize(x)
        x = x.mean(axis=1).detach()
        return x
    
    
def generate_mask(nb_cols=792, nb_rows=1080, nb_channels=3):
    """
    Main mask used to capture the relevant portion of LUS images. Crafted manually. Is [1,1,1] where image is relevant
    """
    mask = np.zeros([nb_rows, nb_cols, nb_channels])

    # Filling mask
    for row in range(nb_rows):
        for col in range(nb_cols):
            # Delimitations of the cone like portion of a LUS image
            if row > 25 and row < 1010 and col < 762 and (-4/5 * row + 293) < col and (4/5*row) + nb_cols-293 > col:
                mask[row, col] = [1,1,1]

    mask = mask.astype('uint8')

    return mask

def prepare_for_model(img, mask):
    """
    Read image and apply required transformations for model usage
    """
    # Read, resize and crop
    img = resize_crop(img, mask)
    # Transform to PIL format
    img = Image.fromarray(img)
    # Model transformations

    # Images transformations to apply before entering model
    transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return transform(img)


def resize_crop(image, mask, nb_cols=792, nb_rows=1080):
    """
    Resize image and apply mask
    """
    masked_img = cv2.resize(image, (nb_cols, nb_rows))*mask
    return masked_img
    
    
    