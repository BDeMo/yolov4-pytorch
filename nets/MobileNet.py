import torch

from torch import nn
import torch.nn.functional as F


class Block(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d \
            (in_planes, in_planes, kernel_size=3, stride=stride,
             padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d \
            (in_planes, out_planes, kernel_size=1,
             stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    cfg1 = [64, (128, 2), 128, (256, 2), (256, 2)]
    cfg2 = [(512, 2), 512, 512, 512, 512, 512, ]
    cfg3 = [(1024, 2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.pre = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU())
        self.stages = nn.ModuleList([
            self._make_layers(self.cfg1, in_planes=32),
            self._make_layers(self.cfg2, in_planes=256),
            self._make_layers(self.cfg3, in_planes=512)
        ])
        self.linear = nn.Linear(1024 * 6 * 6, num_classes)

    def _make_layers(self, cfg, in_planes):
        layers = []
        for x in cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        out1 = self.stages[0](x)
        out2 = self.stages[1](out1)
        out3 = self.stages[2](out2)
        x = F.avg_pool2d(out3, 2)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x, out1, out2, out3


def mobilenet(pretrained):
    model = MobileNet(num_classes=10)
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("mobilenet request a pretrained path. got [{}]".format(pretrained))
    return model


if __name__ == "__main__":
    model = mobilenet(None)
    x = torch.randn(1, 3, 416, 416)
    _, out1, out2, out3 = model(x)
    print(_.size(), out1.shape, out2.shape, out3.shape)
