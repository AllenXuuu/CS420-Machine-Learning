import torch
import torch.nn as nn
from .UNet import UNetBlock


class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, activation='sigmoid'):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.activation = {
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
        }[activation]()

    def forward(self, x):
        return x * self.activation(self.conv(x))


class Network(nn.Module):
    def __init__(self, args, in_channels, out_classes):
        super(Network, self).__init__()
        self.hidden_channles = [64, 128, 256, 512, 1024]
        # self.hidden_channles = [a//2 for a in self.hidden_channles]

        self.maxpooing = nn.MaxPool2d(kernel_size=2)
        self.downBlockList = nn.ModuleList([
            UNetBlock(in_channels, self.hidden_channles[0]),
            UNetBlock(self.hidden_channles[0], self.hidden_channles[1]),
            UNetBlock(self.hidden_channles[1], self.hidden_channles[2]),
            UNetBlock(self.hidden_channles[2], self.hidden_channles[3])
        ])
        self.midBlock = UNetBlock(self.hidden_channles[3], self.hidden_channles[4])
        self.upBlockList = nn.ModuleList([
            UNetBlock(2 * self.hidden_channles[3], self.hidden_channles[3]),
            UNetBlock(2 * self.hidden_channles[2], self.hidden_channles[2]),
            UNetBlock(2 * self.hidden_channles[1], self.hidden_channles[1]),
            UNetBlock(2 * self.hidden_channles[0], self.hidden_channles[0])
        ])
        self.upConvList = nn.ModuleList([
            nn.ConvTranspose2d(self.hidden_channles[4], self.hidden_channles[3], kernel_size=2, stride=2, padding=0),
            nn.ConvTranspose2d(self.hidden_channles[3], self.hidden_channles[2], kernel_size=2, stride=2, padding=0),
            nn.ConvTranspose2d(self.hidden_channles[2], self.hidden_channles[1], kernel_size=2, stride=2, padding=0),
            nn.ConvTranspose2d(self.hidden_channles[1], self.hidden_channles[0], kernel_size=2, stride=2, padding=0),
        ])
        self.attentionList = nn.ModuleList([
            AttentionModule(self.hidden_channles[3], self.hidden_channles[3]),
            AttentionModule(self.hidden_channles[2], self.hidden_channles[2]),
            AttentionModule(self.hidden_channles[1], self.hidden_channles[1]),
            AttentionModule(self.hidden_channles[0], self.hidden_channles[0]),
        ])

        self.classifier = nn.Conv2d(self.hidden_channles[0], out_classes, kernel_size=1, stride=1)

    def forward(self, x):
        memory = []

        for module in self.downBlockList:
            mem = module(x)
            memory.append(mem)
            x = self.maxpooing(mem)

        x = self.midBlock(x)

        for mem, upBlock, upConv, atten in zip(memory[::-1], self.upBlockList, self.upConvList,self.attentionList):
            x = upConv(x)
            x = torch.cat([x, atten(mem)], dim=1)
            x = upBlock(x)

        score = self.classifier(x)
        return score
