'''
GHANMI Helmi, November 2018
version: 1.0 
1-Please create a folder named 'images' and put the input image in this folder
the input mage should be have the same size. Beside you can use the algorithme named 'resize image' 

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

#The content loss function

""" define a fake backward method that calls the backward method of nn.MSELoss"""
class ContentLoss(nn.Module):

	# def the constructor method 
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    # def the forward method 
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

#Style Loss

"""For the style loss we define a module that computes the gram produced given the feature maps of the neural networks. """
def gram_matrix(input):
    a, b, c, d = input.size()  

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t()) 
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


#Loading the neural network
cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
    	return (img - self.mean) / self.std

print('OK OK')
# TODO / ADD THE Loading the neural network
