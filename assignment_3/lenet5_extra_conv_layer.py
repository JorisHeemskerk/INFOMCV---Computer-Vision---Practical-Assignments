from torch import nn

from lenet5_more_feature_kernels import LeNet5MoreFeatureKernels


class LeNet5ExtraConvLayer(LeNet5MoreFeatureKernels):
    """
    Overridden version of the LeNet-5 base model, where there is an
    extra 1x1 layer added before the last conv layer.
    """
    def __init__(
        self,
        n_classes: int,
        n_first_layer_kernels: int,
        n_channels: int
    ):
        """
        Constructs a `LeNet5MoreFeatureKernels` model, then adds a
        convolutional layer before the last convolutional layer.

        :param n_classes: Number of output classes
        :type: n_classes: int
        :param n_first_layer_kernels: Number of kernels in the first 
            layer.
        :type n_first_layer_kernels: int
        """
        assert n_channels > 0, \
            "Cannot have negative number of channels!"
        super().__init__(n_classes, n_first_layer_kernels)
        self.embedding.insert(6, nn.Conv2d(16, n_channels, 1))
        self.embedding.insert(7, nn.ReLU())

        self.embedding.pop(8)
        self.embedding.insert(8, nn.Conv2d(n_channels, 120, 5))
