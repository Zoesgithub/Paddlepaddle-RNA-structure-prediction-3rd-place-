import torch
import torch.nn as nn
import numpy as np
import h5py
from sklearn import metrics
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import operator
from functools import reduce
from loguru import logger
import random
from torch.utils.data import DataLoader


def conv3x3(inc, outc ,ks, stride=1, groups=1, dilation=1):
    return nn.Conv1d(inc, outc, ks, stride, padding=((ks-1)*dilation)//2, groups=groups, bias=True, dilation=dilation)

def conv1x1(inc, outc, stride=1):
    return nn.Conv1d(inc, outc, 1, stride, bias=False)

class BasicBlock(nn.Module):
    expansion=1
    def __init__(self, inc, channel, ks, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer=nn.BatchNorm1d
        self.conv=conv3x3(inc, channel,ks, stride)
        self.bn1=norm_layer(channel)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=conv3x3(channel, channel, ks)
        self.bn2=norm_layer(channel)
        self.downsample=downsample
        self.stride=stride
    def forward(self, inp):
        x=self.conv1(inp)
        x=self.bn1(x)
        x=self.relu(x)

        x=self.conv2(x)
        x=self.bn2(x)
        if self.downsample is not None:
            inp=self.downsample(inp)
        x+=inp
        x=self.relu(x)
        return x


class Bottleneck(nn.Module):
    expansion=4
    def __init__(self, inc, channel, ks, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer=nn.BatchNorm1d

        width=int(channel*base_width//64)*groups
        self.conv1=conv1x1(inc, width)
        self.bn1=norm_layer(width)
        self.conv2=conv3x3(width, width, ks, stride, groups, dilation)
        self.bn2=norm_layer(width)
        self.conv3=conv1x1(width, channel*self.expansion)
        self.bn3=norm_layer(channel*self.expansion)
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample
        self.stride=stride

    def forward(self, inp):
        x=self.conv1(inp)
        x=self.bn1(x)
        x=self.relu(x)

        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)

        x=self.conv3(x)
        x=self.bn3(x)

        if self.downsample is not None:
            inp=self.downsample(inp)
        x+=inp
        x=self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        norm_layer=nn.BatchNorm1d
        self.norm_layer=norm_layer
        self.inc=32
        self.dilation=9
        self.base_width=64
        self.net=[]
        self.net.append(nn.Conv1d(7, self.inc, kernel_size=1, stride=1, padding=0,bias=False))
        self.net.append(nn.Conv1d(self.inc, self.inc, kernel_size=1, stride=1, padding=0))
        self.net.append(norm_layer(self.inc))
        self.net.append(nn.ReLU(inplace=True))
        self.net.append(self._make_layer(Bottleneck, 32, 11, 3, 1, 1))
        self.net.append(self._make_layer(Bottleneck, 32, 11,  3, 1, 6))
        self.net.append(self._make_layer(Bottleneck, 32, 21, 3, 1, 11))
        self.net.append(self._make_layer(Bottleneck, 32,81, 3, 1, 21))
        self.net.append(self._make_layer(Bottleneck, 32,81, 3, 1, 41))
        self.net.append(nn.Dropout(0.5))
        self.Net=nn.Sequential(*self.net)

    def forward(self, x):
        return self.Net(x)
    
    def _make_layer(self, block, channel, ks, blocks, stride=1, dilation=1):
        norm_layer=self.norm_layer
        downsample=None
        if stride!=1 or self.inc!=channel*block.expansion:
            downsample=nn.Sequential(conv1x1(self.inc, channel*block.expansion, stride), norm_layer(channel*block.expansion))
        layers=[]
        layers.append(block(self.inc, channel, ks, stride, downsample, 1, self.base_width, dilation, norm_layer))
        self.inc=channel*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inc, channel, ks, base_width=self.base_width, dilation=dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

class Model(nn.Module):
    def __init__(self, lr):
        super(Model, self).__init__()
        self.Gn=5

        self.cresnet=ResNet()
        self.sig=nn.Sigmoid()
        self.relu=nn.ReLU()
        self.soft=nn.Softmax(-1)
        self.out_fc=nn.Linear(self.cresnet.inc, 1)
        self.cinc=self.cresnet.inc
        self.cresnet=nn.DataParallel(self.cresnet)
        self.ene_fc=nn.Linear(self.cinc, 1)
    
        self.tanh=nn.Tanh()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)


    def forward(self, inp):
        x=self.cresnet(inp)
        out=self.out_fc(x.permute(0,2,1)).squeeze(-1)
        ene=self.ene_fc(x.mean(2)).squeeze(-1)
        return out, ene


    def train_onestep(self, data):
        self.train()
        self.optimizer.zero_grad()
        id, x,y, Ene=data
        Ene=Ene.cuda()

        x=x.permute(0, 2, 1).cuda()
        y=y.cuda()
        out, ene=self.forward(x)
        enemask=(Ene>-9990).float()
        out=self.sig(out)
        loss=(((y-out)**2).sum(1).add(1e-9).sqrt()*(1-enemask)).sum()+((ene-Ene)**2*enemask).mean()
        loss.backward()
        self.optimizer.step()
        return loss 

    def Eval(self, dataloader):
        self.eval()
        acc=0.0
        size=0.0
        rmsd=0.0
        ret=[]
        for data in dataloader:
            id, x, y, _=data
            x = x.permute(0, 2, 1).cuda()
            with torch.no_grad():
                out=self.sig(self.forward(x)[0])
            for i,d in enumerate(id):
                ret.append([d, out[i].detach().cpu().numpy()])
            rmsd+=((out-y.cuda())**2).mean(1).sqrt().sum()
            size+=len(x)

        acc=0#acc/size
        rmsd=rmsd/size
        return acc, rmsd, ret
