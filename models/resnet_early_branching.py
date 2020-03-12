'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        x.requires_grad = True
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, fine_tune=False, classes=(10, 10), pool=True, pretrained=False, size_factor=1, branching=True):
        print(size_factor)
        # print("Fine tune? ", fine_tune)
        super(ResNet, self).__init__()
        if branching:
            num_branches = len(classes)
        else:
            num_branches = 1
        self.num_branches = num_branches
        self.in_planes = 64
        self.pool = pool
        self.classes = classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3_1 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4_1 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.pool_fc_1 = nn.Linear(8192, 512*block.expansion)


        if self.num_branches > 1:
            self.layer3_2 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4_2 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.pool_fc_2 = nn.Linear(8192, 512*block.expansion)
        if self.num_branches > 2:
            self.layer3_3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4_3 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.pool_fc_3 = nn.Linear(8192, 512*block.expansion)
        if self.num_branches == 4:
            self.layer3_4 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4_4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.pool_fc_4 = nn.Linear(8192, 512*block.expansion)

        self.fc2_number = nn.Linear(512*block.expansion, classes[0])
        self.fc2_color = nn.Linear(512*block.expansion, classes[1])
        if len(classes) > 2:
            self.fc2_loc = nn.Linear(512*block.expansion, classes[2])
        if len(classes) == 4:
            self.fc2_scale = nn.Linear(512*block.expansion, classes[3])

        if pool:
            if not pretrained:
                self.name = "resnet"
            else:
                if fine_tune:
                    self.name = "resnet_pretrained"
                else:
                    self.name = "resnet_pretrained_embeddings"
        else:
            self.name = "resnet_no_pool"

        if not fine_tune:
            for p in self.conv1.parameters():
                p.requires_grad = False
            for p in self.bn1.parameters():
                p.requires_grad = False
            for p in self.layer1.parameters():
                p.requires_grad = False
            for p in self.layer2.parameters():
                p.requires_grad = False
            for p in self.layer3_1.parameters():
                p.requires_grad = False
            for p in self.layer4_1.parameters():
                p.requires_grad = False
            if self.num_branches > 1:
                for p in self.layer3_2.parameters():
                    p.requires_grad = False
                for p in self.layer4_2.parameters():
                    p.requires_grad = False
            if self.num_branches > 2:
                for p in self.layer3_3.parameters():
                    p.requires_grad = False
                for p in self.layer4_3.parameters():
                    p.requires_grad = False
            if self.num_branches == 4:
                for p in self.layer3_4.parameters():
                    p.requires_grad = False
                for p in self.layer4_4.parameters():
                    p.requires_grad = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        if planes == 64:
            self.in_planes = 64
        else:
            self.in_planes = planes//2
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out1 = self.layer3_1(out)
        out1 = self.layer4_1(out1)
        if self.pool:
            out1 = F.avg_pool2d(out1, 4)
        else:
            out1 = out1.view(out1.size(0), -1)
            out1 = self.pool_fc_1(out1)
            out1 = F.relu(out1)
        out1 = out1.view(out1.size(0), -1)

        if self.num_branches == 1:
            num = self.fc2_number(out1)
            col = self.fc2_color(out1)
            if len(self.classes) == 3:
                loc = self.fc2_loc(out1)
                return F.log_softmax(num, dim=1), F.log_softmax(col, dim=1), F.log_softmax(loc, dim=1)
            elif len(self.classes) == 4:
                loc = self.fc2_loc(out1)
                scale = self.fc2_scale(out1)
                return F.log_softmax(num, dim=1), F.log_softmax(col, dim=1), F.log_softmax(loc, dim=1), F.log_softmax(scale, dim=1)
            else:
                return F.log_softmax(num, dim=1), F.log_softmax(col, dim=1)

        if self.num_branches > 1:
            out2 = self.layer3_2(out)
            out2 = self.layer4_2(out2)
            if self.pool:
                out2 = F.avg_pool2d(out2,4)
            else:
                out2 = out2.view(out2.size(0),-1)
                out2 = self.pool_fc_2(out2)
                out2 = F.relu(out2)
            out2 = out2.view(out2.size(0), -1)
            num = self.fc2_number(out1)
            col = self.fc2_color(out2)

        if self.num_branches > 2:
            out3 = self.layer3_3(out)
            out3 = self.layer4_3(out3)
            if self.pool:
                out3 = F.avg_pool2d(out3)
            else:
                out3 = out3.view(out3.size(0), -1)
                out3 = self.pool_fc_3(out3)
                out3 = F.relu(out3)
            out3 = out3.view(out3.size(0), -1)
            loc = self.fc2_loc(out3)

        if self.num_branches == 4:
            out4 = self.layer3_4(out)
            out4 = self.layer4_4(out4)
            if self.pool:
                out4 = F.avg_pool2d(out4)
            else:
                out4 = out4.view(out4.size(0),-1)
                out4 = self.pool_fc_4(out4)
                out4 = F.relu(out4)
            out4 = out4.view(out4.size(0), -1)
            scale = self.fc2_scale(out4)

        if len(self.classes) == 3:
            return F.log_softmax(num, dim=1), F.log_softmax(col, dim=1), F.log_softmax(loc, dim=1)
        elif len(self.classes) == 4:
            return F.log_softmax(num, dim=1), F.log_softmax(col, dim=1), F.log_softmax(loc, dim=1), F.log_softmax(scale, dim=1)
        else:
            return F.log_softmax(num, dim=1), F.log_softmax(col, dim=1)


def ResNet18(pretrained=False, fine_tune=False, classes=(10, 10), pool=True, branching=True):
    # print("Pretrained? ", pretrained)
    model = ResNet(BasicBlock, [2, 2, 2, 2], fine_tune=fine_tune, classes=classes, pool=pool, pretrained=pretrained, branching=branching)

    if not pretrained:
        return model

    model_dict = model.state_dict()

    # original saved file with DataParallel
    state_dict = torch.load('models/st_dict_epoch71.pt', map_location="cpu")
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        # print(name, type(name))
        if name in model_dict:
            new_state_dict[name] = v

    # overwrite entries in the existing state dict
    model_dict.update(new_state_dict)
    # load the new state dict
    model.load_state_dict(model_dict)
    return model


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
