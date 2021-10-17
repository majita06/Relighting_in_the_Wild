from torchvision import models
import torch.nn as nn

class VGG19(nn.Module):
    """docstring for Vgg19"""
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(4): # conv1_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9): # conv2_2
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 14): # conv3_2
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 23): # conv4_2
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 32): # conv5_2
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_2 = h
        h = self.slice4(h)
        h_relu4_2 = h
        h = self.slice5(h)
        h_relu5_2 = h

        out = {}
        out['conv1_2'] = h_relu1_2
        out['conv2_2'] = h_relu2_2
        out['conv3_2'] = h_relu3_2
        out['conv4_2'] = h_relu4_2
        out['conv5_2'] = h_relu5_2

        return out

