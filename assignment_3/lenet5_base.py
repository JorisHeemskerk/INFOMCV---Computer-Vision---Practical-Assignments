import torch
from torch import nn
import torch.nn.functional as F


class LeNet5Base(nn.Module):
    """
    The LeNet-5 model architecture but for color images.
    """
    def __init__(self, n_classes: int)-> None:
        """
        Define the convolutional, pooling and fully connected layers.

        :param n_classes: Number of output classes
        :type: n_classes: int
        """
        super(LeNet5Base, self).__init__()
        self.embedding = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, 5),
            nn.ReLU()
        )
        
        self.fully_connected = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(84, n_classes)
        )

    def initialize_weights(self)-> None:
        """
        Apply kaiming uniform initialization to all layers. This
        function should be called manually after creating a class
        instance.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
                print(module.bias)


    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        Perform a forward pass on the network.

        :param x:
        :type x: torch.Tensor
        :return: Input tensor of shape (batch_size, 3, 32, 32).
        """
        x = self.embed(x)
        x = self.head(x)
        return F.softmax(x, dim=1)
    
    def embed(self, x: torch.Tensor)-> torch.Tensor:
        """
        This method only embeds the data using the 6th fully connected
        layer. It does not make predictions.

        :param x: Input tensor.
        :type x: torch.Tensor
        :returns: Embedded tensor of length 84.
        :rtype: torch.Tensor
        """
        x = self.embedding(x)
        x = torch.flatten(x, 1)
        return self.fully_connected(x)

    def save(self, dir: str)-> None:
        """
        Save internal state to file.

        :param dir: Directory to output model to.
        :type dir: str
        """
        filename = f"{dir}/best_{self.__class__.__name__}.pth"
        print(f"\033[36mSaving model...\033[37m")
        torch.save(self, filename)

