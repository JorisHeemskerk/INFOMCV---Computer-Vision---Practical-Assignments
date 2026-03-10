from torch import nn

from lenet5_base import LeNet5Base


class LeNet5MoreFeatureKernels(LeNet5Base):
    """
    Overridden version of the LeNet-5 base model, where the number of 
    kernels in the first layer are increased.
    """
    def __init__(self, n_first_layer_kernels: int):
        """
        Constructs a `LetNet5Base` model, then replaces the first 
        convolutional layer to have a different number of kernels.

        :param n_first_layer_kernels: Number of kernels in the first 
            layer.
        :type n_first_layer_kernels: int
        """
        assert n_first_layer_kernels > 0, \
            "Cannot have negative number of kernels!"
        super(LeNet5MoreFeatureKernels, self).__init__()
        self.embedding.pop(0)
        self.embedding.insert(0, nn.Conv2d(3, n_first_layer_kernels, 5))

        self.embedding.pop(3)
        self.embedding.insert(3, nn.Conv2d(n_first_layer_kernels, 16, 5))
