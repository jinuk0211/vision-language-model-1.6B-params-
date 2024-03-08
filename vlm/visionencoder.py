import torch
from einops import rearrange
from PIL import Image
from torch import nn
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    InterpolationMode,
    ToImage,
    ToDtype,
    Normalize,
)
import timm

class visionholder(nn.Module):
    def __init__(self,model):
        super().__init__():
        self.visual = model

    def forward(self,x): #visionencoder 
        return self.visual(x)

class modelholder(nn.Module):
    def __init__(self,model):
        super().__init():
        self.model = model
    def forward(self,x): #model transformer
        return self.model(x)
        
class linearpatchembedding(nn.Module):
    def __init__(self,conv):
        super().__init__():
        self.linear = nn.Linear(588,1152)
        self.linear.weight.data = conv.weight.data.view(1152,-1)
        if conv.bias is not None:
            self.linear.bias.data = conv.bias.data
    def forward(self,x):
        return self.linear(x)

class MLP(nn.module): #다중 퍼셉트론 layer
    def __init__(self,
                 infeatures,
                 hiddenfeatures,
                 outfeatures,
                 activation : nn.Module = nn.GELU,
                ) -> None:
        super().__init__()
                    
    
class visionencoder
