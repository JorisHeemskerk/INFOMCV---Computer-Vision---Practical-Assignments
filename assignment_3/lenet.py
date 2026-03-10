import torch
from torch import nn


class LeNet5(nn.Module):
    """
    The LeNet-5 model architecture but for color images.
    """
    def __init__(self)-> None:
        """
        Define the convolutional, pooling and fully connected layers.
        """
        super(LeNet5, self).__init__()
        self.embedding = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 14, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(14, 120, 5),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU()
        )
        # self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv3 = nn.Conv2d(6, 16, 5, padding=2)
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv5 = nn.Conv2d(16, 120, 5, padding=2)

        self.head = nn.Sequential(
            nn.Linear(84, 10),
            nn.softmax(dim=1)
        )
        # self.fc6 = nn.Linear(7680, 84)
        # self.fc7 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        Perform a forward pass on the network.

        :param x:
        :type x: torch.Tensor
        :return: Input tensor of shape (batch_size, 3, 32, 32).
        """
        x = self.embedding(x)
        x = self.head(x)
        return x
    
    # def embed(self, x: torch.Tensor):
    #     """
    #     This method only embeds the data using the 6th fully connected
    #     layer. It does not make predictions.

    #     :param x: Input tensor.
    #     :type x: torch.Tensor
    #     :returns: Embedded tensor of length 84.
    #     :rtype: torch.Tensor
    #     """
    #     x = self.conv1(x)
    #     x = F.relu(x)
    #     x = self.pool2(x)
    #     x = self.conv3(x)
    #     x = F.relu(x)
    #     x = self.pool4(x)
    #     x = self.conv5(x)
    #     x = F.relu(x)
    #     x = torch.flatten(x, 1)
    #     x = self.fc6(x)
    #     return F.relu(x)
