import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layer=2, stride=1, kernel_size=3, padding=1, batchnorm=True):
        super(UNetBlock, self).__init__()

        self.layers = []

        for _ in range(num_layer):
            self.layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding)
            )
            if batchnorm:
                self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU())
            in_channels = out_channels

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


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

        self.classifier = nn.Conv2d(self.hidden_channles[0], out_classes, kernel_size=1, stride=1)

    def forward(self, x):
        memory = []

        for module in self.downBlockList:
            mem = module(x)
            memory.append(mem)
            x = self.maxpooing(mem)

        x = self.midBlock(x)

        for mem, upBlock, upConv in zip(memory[::-1], self.upBlockList, self.upConvList):
            x = upConv(x)
            x = torch.cat([x, mem], dim=1)
            x = upBlock(x)

        score = self.classifier(x)
        return score
