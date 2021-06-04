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
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),  # 这个卷积操作是不会改变w h的
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

        # 重复layer，每个layer都包含多个残差块 其中第一个残差会修改w和c，其他的残差块等量变换
        # 经过第一个残差块后大小为 w-1/s +1 （每个残差块包括left和right，而left的k = 3 p = 1，right的shortcut k=1，p=0）
        self.stages = nn.ModuleList([
            self._make_layer(64, 128, 3),  # s默认是1 ,所以经过layer1后只有channle变了
            self._make_layer(128, 256, 4, stride=2),  # w-1/s +1
            self._make_layer(256, 512, 6, stride=2),
            self._make_layer(512, 1024, 3, stride=2)
        ])
        self.fc = nn.Linear(1024, num_class)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        # 刚开始两个cahnnle可能不同，所以right通过shortcut把通道也变为outchannel
        shortcut = nn.Sequential(
            # 之所以这里的k = 1是因为，我们在ResidualBlock中的k =3,p=1所以最后得到的大小为(w+2-3/s +1)
            # 即(w-1 /s +1)，而这里的w = (w +2p-f)/s +1 所以2p -f = -1 如果p = 0则f = 1
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        # 之后的cahnnle同并且 w h也同，而经过ResidualBloc其w h不变，
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.pre(input)

        x = self.stages[0](x)
        out1 = self.stages[1](x)
        out2 = self.stages[2](out1)
        out3 = self.stages[3](out2)
        x = F.avg_pool2d(out3, 7)  # 如果图片大小为224 ，经过多个ResidualBlock到这里刚好为7，所以做一个池化，为1，
        # 所以如果图片大小小于224，都可以传入的，因为经过7的池化，肯定为1，但是大于224则不一定
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, out1, out2, out3
        #now out1-3:
        # torch.Size([1, 256, 52, 52]) torch.Size([1, 512, 26, 26]) torch.Size([1, 512, 13, 13])
        #target out1-3:
        # torch.Size([1, 256, 52, 52]) torch.Size([1, 512, 26, 26]) torch.Size([1, 1024, 13, 13])
    #DONE

def imshow(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))

def resnet17(pretrained):
    model = ResNet()
    if pretrained:
        if isinstance(pretrained, str):
            model_dict = model.state_dict()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained, map_location=device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            raise Exception("resnet request a pretrained path. got [{}]".format(pretrained))
    return model

if __name__ == "__main__":
    model = resnet17(None)
    x = torch.randn(1, 3, 416, 416)
    _, out1, out2, out3 = model(x)
    print(_.shape, out1.shape, out2.shape, out3.shape)