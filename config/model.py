import models.simple_CNN
import models.resnet

options = {
    "simple_cnn": lambda num_classes: models.simple_CNN.SimpleCNN(num_classes),
    "resnet": lambda num_classes: models.resnet.ResNet18(pretrained=False, fine_tune=True, classes=num_classes),
    "resnet_pretrained": lambda num_classes: models.resnet.ResNet18(pretrained=True, fine_tune=True, classes=num_classes),
    "resnet_pretrained_embeddings": lambda num_classes: models.resnet.ResNet18(pretrained=True, fine_tune=False, classes=num_classes)
}
