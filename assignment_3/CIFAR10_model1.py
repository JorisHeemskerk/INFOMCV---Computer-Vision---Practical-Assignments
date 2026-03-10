from torch import nn

from lenet import LeNet5


class CIFAR10_model1(LeNet5):
    def __init__(self):
        super().__init__()
        self.embedding.pop(0)
        self.embedding.insert(0, nn.Conv2d(3, 12, 5))

        self.embedding.pop(1)
        self.embedding.insert(1, nn.Conv2d(12, 16, 5))