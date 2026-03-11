from torch import nn

from lenet5_more_feature_kernels import LeNet5MoreFeatureKernels


class LeNet5ExtraConvLayer(LeNet5MoreFeatureKernels):
    """
    Overridden version of the LeNet-5 base model, where there is an
    extra 1x1 layer added before the last conv layer.
    """
    def __init__(self, n_channels: int):
        """
        Constructs a `LetNet5Base` model, then replaces the first 
        convolutional layer to have a different number of kernels.

        :param n_first_layer_kernels: Number of kernels in the first 
            layer.
        :type n_first_layer_kernels: int
        """
        assert n_channels > 0, \
            "Cannot have negative number of channels!"
        super().__init__()
        self.embedding.insert(6, nn.Conv2d(16, n_channels, 1))
        self.embedding.insert(7, nn.ReLU())

        self.embedding.pop(8)
        self.embedding.insert(8, nn.Conv2d(n_channels, 120, 5))
