import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, input):
        out = self.left(input)
        residual = input if self.right is None else self.right(input)
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, num_class=1000):
        super().__init__()
        # 前面几层普通卷积
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.stages = nn.ModuleList([
            self._make_layer(64, 128, 3),
            self._make_layer(128, 256, 4, stride=2),
            self._make_layer(256, 512, 6, stride=2),
            self._make_layer(512, 1024, 3, stride=2)
        ])
        self.fc = nn.Linear(1024, num_class)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.pre(input)

        x = self.stages[0](x)
        out1 = self.stages[1](x)
        out2 = self.stages[2](out1)
        out3 = self.stages[3](out2)
        x = F.avg_pool2d(out3, 7)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, out1, out2, out3


def resnet17(pretrained):
    model = ResNet(num_class=10)
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("resnet request a pretrained path. got [{}]".format(pretrained))
    return model


def imshow(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))


if __name__ == "__main__":
    model = resnet17(None)
    x = torch.randn(1, 3, 416, 416)
    _, out1, out2, out3 = model(x)
    print(_.shape, out1.shape, out2.shape, out3.shape)
