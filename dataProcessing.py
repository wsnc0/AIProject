import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# import numpy as np
import matplotlib.pyplot as plt

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])


def pad_to_Square(image):
    width, height = image.size
    max_side = max(width,height)
    left_pad = (max_side-width)//2
    right_pad = (max_side-width-left_pad)
    top_pad = (max_side-height)//2
    bot_pad = (max_side-height-top_pad)

    padded_image = transforms.functional.pad(image, (left_pad, top_pad, right_pad, bot_pad), padding_mode='constant', fill=0)
    return padded_image

transform = transforms.Compose([
    transforms.Lambda(pad_to_Square),
    transforms.Resize((256, 256)),  # Resize images
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize(mean=mean, std=std),  # Normalize images
])


path = './data'
# Load dataset from directory
dataset = datasets.ImageFolder(root=path, transform=transform)
print(dataset, len(dataset))
# Create a DataLoader
dataloader_main = DataLoader(dataset, shuffle=True)
# dataloader_main = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
print(len(dataloader_main))
print(dataloader_main)