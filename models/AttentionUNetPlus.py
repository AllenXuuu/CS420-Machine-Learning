import torch
import torch.nn as nn
from .UNet import UNetBlock
from .AttentionUNet import Attention


class NestBlock(nn.Module):
    def __init__(self, in_channel, out_channel, n_feature=2):
        super(NestBlock, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channel, out_channel, kernel_size=2, stride=2, padding=0)
        self.conv = UNetBlock(out_channel * n_feature, out_channel)
        self.attenList = nn.ModuleList([
            Attention(out_channel) for i in range(n_feature - 1)
        ])

    def forward(self, high, *low):
        out = self.up(high)
        for feat, atten in zip(low, self.attenList):
            out = torch.cat([out, atten(feat)], 1)
        return self.conv(out)


class Network(nn.Module):
    def __init__(self, args, in_channels, out_classes):
        super(Network, self).__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.hidden_channles = [64, 128, 256, 512, 1024]

        ## Encoder
        self.maxpooing = nn.MaxPool2d(kernel_size=2)
        self.conv00 = UNetBlock(self.in_channels, self.hidden_channles[0])
        self.conv10 = UNetBlock(self.hidden_channles[0], self.hidden_channles[1])
        self.conv20 = UNetBlock(self.hidden_channles[1], self.hidden_channles[2])
        self.conv30 = UNetBlock(self.hidden_channles[2], self.hidden_channles[3])
        self.conv40 = UNetBlock(self.hidden_channles[3], self.hidden_channles[4])

        ## Decoder
        self.up01 = NestBlock(self.hidden_channles[1], self.hidden_channles[0], n_feature=2)
        self.up11 = NestBlock(self.hidden_channles[2], self.hidden_channles[1], n_feature=2)
        self.up21 = NestBlock(self.hidden_channles[3], self.hidden_channles[2], n_feature=2)
        self.up31 = NestBlock(self.hidden_channles[4], self.hidden_channles[3], n_feature=2)

        self.up02 = NestBlock(self.hidden_channles[1], self.hidden_channles[0], n_feature=3)
        self.up12 = NestBlock(self.hidden_channles[2], self.hidden_channles[1], n_feature=3)
        self.up22 = NestBlock(self.hidden_channles[3], self.hidden_channles[2], n_feature=3)

        self.up03 = NestBlock(self.hidden_channles[1], self.hidden_channles[0], n_feature=4)
        self.up13 = NestBlock(self.hidden_channles[2], self.hidden_channles[1], n_feature=4)

        self.up04 = NestBlock(self.hidden_channles[1], self.hidden_channles[0], n_feature=5)

        ## classifier
        self.classifier_1 = nn.Conv2d(self.hidden_channles[0], self.out_classes, kernel_size=1)
        self.classifier_2 = nn.Conv2d(self.hidden_channles[0], self.out_classes, kernel_size=1)
        self.classifier_3 = nn.Conv2d(self.hidden_channles[0], self.out_classes, kernel_size=1)
        self.classifier_4 = nn.Conv2d(self.hidden_channles[0], self.out_classes, kernel_size=1)

    def forward(self, x):
        X_00 = self.conv00(x)
        X_10 = self.conv10(self.maxpooing(X_00))
        X_20 = self.conv20(self.maxpooing(X_10))
        X_30 = self.conv30(self.maxpooing(X_20))
        X_40 = self.conv40(self.maxpooing(X_30))

        # column : 1
        X_01 = self.up01(X_10, X_00)
        X_11 = self.up11(X_20, X_10)
        X_21 = self.up21(X_30, X_20)
        X_31 = self.up31(X_40, X_30)

        # column : 2
        X_02 = self.up02(X_11, X_00, X_01)
        X_12 = self.up12(X_21, X_10, X_11)
        X_22 = self.up22(X_31, X_20, X_21)

        # column : 3
        X_03 = self.up03(X_12, X_00, X_01, X_02)
        X_13 = self.up13(X_22, X_10, X_11, X_12)

        # column : 4
        X_04 = self.up04(X_13, X_00, X_01, X_02, X_03)

        score_1 = self.classifier_1(X_01)
        score_2 = self.classifier_2(X_02)
        score_3 = self.classifier_3(X_03)
        score_4 = self.classifier_4(X_04)

        score = (score_1 + score_2 + score_3 + score_4) / 4
        return score
