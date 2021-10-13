
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np

train_baseline = True

class ResidualBlock(nn.Module):
    def __init__(self, n_in, n_out, stride=1, ksize=3):
        w = math.sqrt(2)
        super().__init__()

        self.c1=nn.Conv2d(n_in, n_out, ksize, stride=stride, padding = 1)
        nn.init.constant_(self.c1.weight, w)
        self.c2=nn.Conv2d(n_out, n_out, ksize, stride=1, padding = 1)
        nn.init.constant_(self.c2.weight, w)
        self.b1=nn.BatchNorm2d(n_out)
        self.b2=nn.BatchNorm2d(n_out)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return h + x

class CNNAE2ResNet(nn.Module):

    def __init__(self,in_channels=3,train=True):
        super(CNNAE2ResNet,self).__init__()
        self.c0 = nn.Conv2d(in_channels, 64, 4, stride=2, padding=1) # 1024 -> 512
        nn.init.normal_(self.c0.weight, 0.0, 0.02)
        self.c1 = nn.Conv2d(64, 128, 4, stride=2, padding=1,bias=False)  # 512 -> 256
        nn.init.normal_(self.c1.weight, 0.0, 0.02)
        self.c2 = nn.Conv2d(128, 256, 4, stride=2, padding=1,bias=False) # 256 -> 128
        nn.init.normal_(self.c2.weight, 0.0, 0.02)
        self.c3 = nn.Conv2d(256, 512, 4, stride=2, padding=1,bias=False) # 128 -> 64
        nn.init.normal_(self.c3.weight, 0.0, 0.02)
        self.c4 = nn.Conv2d(512, 512, 4, stride=2, padding=1,bias=False) # 64 -> 32
        nn.init.normal_(self.c4.weight, 0.0, 0.02)
        self.c5 = nn.Conv2d(512, 512, 4, stride=2, padding=1,bias=False) # 32 -> 16
        nn.init.normal_(self.c5.weight, 0.0, 0.02)

        self.ra = ResidualBlock(512, 512)
        self.rb = ResidualBlock(512, 512)

        self.dc0a = nn.Conv2d(512,512,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.dc0a.weight, 0.0, 0.02)
        self.up0a = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dc1a = nn.Conv2d(1024, 512,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.dc1a.weight, 0.0, 0.02)
        self.up1a = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dc2a = nn.Conv2d(1024, 256,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.dc2a.weight, 0.0, 0.02)
        self.up2a = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dc3a = nn.Conv2d(512, 128,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.dc3a.weight, 0.0, 0.02)
        self.up3a = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dc4a = nn.Conv2d(256, 64,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.dc4a.weight, 0.0, 0.02)
        self.up4a = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dc5a = nn.Conv2d(128, 9,kernel_size=3,padding=1)
        nn.init.normal_(self.dc5a.weight, 0.0, 0.02)
        self.up5a = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dc0b = nn.Conv2d(512, 512,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.dc0b.weight, 0.0, 0.02)
        self.up0b = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dc1b = nn.Conv2d(1024, 512,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.dc1b.weight, 0.0, 0.02)
        self.up1b = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dc2b = nn.Conv2d(1024, 256,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.dc2b.weight, 0.0, 0.02)
        self.up2b = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dc3b = nn.Conv2d(512, 128,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.dc3b.weight, 0.0, 0.02)
        self.up3b = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dc4b = nn.Conv2d(256, 64,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.dc4b.weight, 0.0, 0.02)
        self.up4b = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dc5b = nn.Conv2d(128, 4,kernel_size=3,padding=1)
        nn.init.normal_(self.dc5b.weight, 0.0, 0.02)
        self.up5b = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

        self.c0l = nn.Conv2d(512*3, 512, 4, stride=2, padding=1,bias=False) # 16 -> 8
        nn.init.normal_(self.c0l.weight, 0.0, 0.02)
        self.c1l = nn.Conv2d(512, 256, 4, stride=2, padding=1,bias=False)  # 8 -> 4
        nn.init.normal_(self.c1l.weight, 0.0, 0.02)
        self.c2l = nn.Conv2d(256, 128, 4, stride=2, padding=1,bias=False) # 4 -> 2
        nn.init.normal_(self.c2l.weight, 0.0, 0.02)
        self.c3l = nn.Conv2d(128, 27, 4, stride=2, padding=1) # 2 -> 1
        nn.init.normal_(self.c3l.weight, 0.0, 0.02)
            
        self.bnc1 = nn.BatchNorm2d(128)
        self.bnc2 = nn.BatchNorm2d(256)
        self.bnc3 = nn.BatchNorm2d(512)
        self.bnc4 = nn.BatchNorm2d(512)
        self.bnc5 = nn.BatchNorm2d(512)

        self.bndc0a = nn.BatchNorm2d(512)
        self.bndc1a = nn.BatchNorm2d(512)
        self.bndc2a = nn.BatchNorm2d(256)
        self.bndc3a = nn.BatchNorm2d(128)
        self.bndc4a = nn.BatchNorm2d(64)

        self.bndc0b = nn.BatchNorm2d(512)
        self.bndc1b = nn.BatchNorm2d(512)
        self.bndc2b = nn.BatchNorm2d(256)
        self.bndc3b = nn.BatchNorm2d(128)
        self.bndc4b = nn.BatchNorm2d(64)
            
        self.bnc0l = nn.BatchNorm2d(512)
        self.bnc1l = nn.BatchNorm2d(256)
        self.bnc2l = nn.BatchNorm2d(128)
        
        self.train_dropout = train


    def forward(self, xi):
        hc0 = F.leaky_relu(self.c0(xi),inplace=True)
        hc1 = F.leaky_relu(self.bnc1(self.c1(hc0)),inplace=True)
        hc2 = F.leaky_relu(self.bnc2(self.c2(hc1)),inplace=True)
        hc3 = F.leaky_relu(self.bnc3(self.c3(hc2)),inplace=True)
        hc4 = F.leaky_relu(self.bnc4(self.c4(hc3)),inplace=True)
        hc5 = F.leaky_relu(self.bnc5(self.c5(hc4)),inplace=True)

        if train_baseline == True:
            hra = self.ra(hc5)

            ha = self.up0a(F.relu(F.dropout(self.bndc0a(self.dc0a(hra)), 0.5, training=self.train_dropout),inplace=True))
            ha = torch.cat((ha,hc4),1)
            ha = self.up1a(F.relu(F.dropout(self.bndc1a(self.dc1a(ha)), 0.5, training=self.train_dropout),inplace=True))
            ha = torch.cat((ha,hc3),1)
            ha = self.up2a(F.relu(F.dropout(self.bndc2a(self.dc2a(ha)), 0.5, training=self.train_dropout),inplace=True))
            ha = torch.cat((ha,hc2),1)
            ha = self.up3a(F.relu(self.bndc3a(self.dc3a(ha)),inplace=True))
            ha = torch.cat((ha,hc1),1)
            ha = self.up4a(F.relu(self.bndc4a(self.dc4a(ha)),inplace=True))
            ha = torch.cat((ha,hc0),1)
            ha = self.up5a(self.dc5a(ha))

        hrb = self.rb(hc5)
        hb = self.up0b(F.relu(F.dropout(self.bndc0b(self.dc0b(hrb)), 0.5, training=self.train_dropout),inplace=True))
        hb = torch.cat((hb,hc4),1)
        hb = self.up1b(F.relu(F.dropout(self.bndc1b(self.dc1b(hb)), 0.5, training=self.train_dropout),inplace=True))
        hb = torch.cat((hb,hc3),1)
        hb = self.up2b(F.relu(F.dropout(self.bndc2b(self.dc2b(hb)), 0.5, training=self.train_dropout),inplace=True))
        hb = torch.cat((hb,hc2),1)
        hb = self.up3b(F.relu(self.bndc3b(self.dc3b(hb)),inplace=True))
        hb = torch.cat((hb,hc1),1)
        hb = self.up4b(F.relu(self.bndc4b(self.dc4b(hb)),inplace=True))
        hb = torch.cat((hb,hc0),1)
        hb = self.up5b(self.dc5b(hb))
        if train_baseline == True:
            hb = self.sig(hb)
        else:
            hb = self.tanh(hb)
        
        if train_baseline == True:
            hc = torch.cat((hc5, hra, hrb),1)
            hc = F.leaky_relu(self.bnc0l(self.c0l(hc)),inplace=True)
            hc = F.leaky_relu(self.bnc1l(self.c1l(hc)),inplace=True)
            hc = F.leaky_relu(self.bnc2l(self.c2l(hc)),inplace=True)
            hc = torch.reshape(self.c3l(hc), (9, 3))
            
        if train_baseline == True:
            return ha, hb, hc
        else:
            return hb