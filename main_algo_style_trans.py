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


style_img = image_loader("images/messi.jpg") # the reference image 
content_img = image_loader("images/resized_stade.jpg")    # the input image should be modified its style 

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

"""plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')
"""
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

# add asequential model of CNN with 5 layers 
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
          
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
            
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

# You can use a different image, but it has to have the same dimension as the other images.

input_img = content_img.clone()
"""plt.figure()
imshow(input_img, title='Input Image')"""

# add an optimizer 
""" L-BFGS algorithm to run our gradient descent."""
""" We create a PyTorch L-BFGS optimizer optim.LBFGS and pass the image as the tensor to optimize. 
We use .requires_grad_() to ensure that the image requires gradient."""

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

#run the backward methods of each loss to dynamically compute their gradients and perform gradient descent.
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
   
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

  
    input_img.data.clamp_(0, 1)

    return input_img

# run the model and show the result (modified output image )
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
#imshow(output, title='Output Image')

plt.ioff()
#plt.save('detection_reult.png')
plt.axis('off')
#number = input('Please entre a number to save the resull with id:')
#plt.savefig("output_image/result"+str(number)+".png", bbox_inches='tight')
plt.savefig("output_image/result.png", bbox_inches='tight')
plt.show()



print('OK the processing is finished !!')
# TODO : optimisation of the code representation 
