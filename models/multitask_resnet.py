import torchvision.models.resnet as resnet
import torch.nn as nn
import torch.nn.functional as F
import torch


# Multitask extension of Resnet
class MultiTaskResNet(resnet.ResNet):
    def __init__(self, fine_tune, classes, pool):
        super(MultiTaskResNet, self).__init__(resnet.BasicBlock, [2, 2, 2, 2])
        self.pool = pool
        self.classes = classes
        if pool:
            self.fc2_number = nn.Linear(512 * resnet.BasicBlock.expansion, classes[0])
            self.fc2_color = nn.Linear(512 * resnet.BasicBlock.expansion, classes[1])
        else:
            self.fc2_number = nn.Linear(2048 * resnet.BasicBlock.expansion, classes[0])
            self.fc2_color = nn.Linear(2048 * resnet.BasicBlock.expansion, classes[1])
        if len(classes) == 3:
            if pool:
                self.fc2_loc = nn.Linear(512 * resnet.BasicBlock.expansion, classes[2])
            else:
                self.fc2_loc = nn.Linear(2048 * resnet.BasicBlock.expansion, classes[2])

        if not fine_tune:
            for p in self.conv1.parameters():
                p.requires_grad = False
            for p in self.bn1.parameters():
                p.requires_grad = False
            for p in self.layer1.parameters():
                p.requires_grad = False
            for p in self.layer2.parameters():
                p.requires_grad = False
            for p in self.layer3.parameters():
                p.requires_grad = False
            for p in self.layer4.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.pool:
            x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        num = self.fc2_number(x)
        col = self.fc2_color(x)

        if len(self.classes) == 2:
            return F.log_softmax(num, dim=1), F.log_softmax(col, dim=1)
        else:
            loc = self.fc2_loc(x)
            return F.log_softmax(num, dim=1), F.log_softmax(col, dim=1), F.log_softmax(loc, dim=1)


def ResNet18(pretrained=False, fine_tune=False, classes=(10, 10), pool=True):
    model = MultiTaskResNet(fine_tune, classes, pool)

    if pretrained:
        state_dict = resnet.load_state_dict_from_url(resnet.model_urls['resnet18'])
        model.load_state_dict(state_dict)

    return model
