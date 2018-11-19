'''
GHANMI Helmi, November 2018
version: 1.0 
1-Please create a folder named 'images' and put the input image in this folder 

'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#1. This is important because neural networks are trained with 0–1 image tensors.

imsize = 512 if torch.cuda.is_available() else 128  

loader = transforms.Compose([
    transforms.Resize(imsize), 
    transforms.ToTensor()])  

def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader("images/apple.jpg")
content_img = image_loader("images/fig.jpg")

assert style_img.size() == content_img.size(), \
"You have to to import style and content images of the same size"

#Displaying the images

"""1-Reconvert the images to PIL images.
    2-Clone the tensor so as to not make changes to it.
    3-Remove the fake batch dimension.
    4-Pause so that plots are updated.
    5-Plot the images using imshow."""

unloader = transforms.ToPILImage() 
plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone() 
    image = image.squeeze(0)      
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(1) 

plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')

# TODO / ADD THE Loading the neural network
