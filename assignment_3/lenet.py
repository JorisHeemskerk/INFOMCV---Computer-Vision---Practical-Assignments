import torch
from torch import nn
import torch.nn.functional as F


class Lenet5(nn.Module):
    """
    The LeNet-5 model architecture but for color images.
    """
    def __init__(self):
        super.__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)