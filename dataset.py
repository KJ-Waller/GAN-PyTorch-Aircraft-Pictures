import torch
import torch.nn as nn
import numpy as np
import torchvision.utils as utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def get_dataloader(batch_size=64, num_workers=0):
    image_folder = './data/'
    # img_size = (64,64)
    img_size = (256,256)
    image_preprocessing = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.ImageFolder(image_folder, image_preprocessing)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)