import torch
import torch.nn as nn
import torchvision.models as models

def load_vgg_model():
    vgg = models.vgg19(pretrained=True).features
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg

def run_style_transfer(content_image, style_image):
    pass