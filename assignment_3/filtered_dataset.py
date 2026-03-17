from torch.utils.data import Dataset, Subset
from typing import Any

class FilteredDataset(Dataset):
    """
    A dataset that contains a filtered and remapped subset of a base
    subset, stripping away the unused data in the subset.
    """
    def __init__(
        self, 
        subset: Subset, 
        index_remap: dict[int, int],
        classes: list[str]
    ):
        """
        Initialise and filter the dataset to strip away all unused data.

        :param subset: Input subset containing extra data.
        :type subset: Subset
        :param index_remap: Dictionary allowing for remapping of target
            labels.
        :type index_remap: dict[int, int]
        :param classes: The classes the labels map to. 
        :type classes: list[str]
        """
        self.classes = classes

        # Filter out images that should be kept according to index_remap
        self.data = [
            (img, index_remap[label]) 
            for img, label in (subset.dataset[i] for i in subset.indices)
            if label in index_remap
        ]

    def __len__(self)-> int:
        """
        Gets the number of datapoints in the dataset.

        :returns: Number of datapoints.
        :rtype: int
        """
        return len(self.data)

    def __getitem__(self, index: int)-> tuple[Any, int]:
        """
        Return the datapoint at the given index.
        :param index: Index of the datapoint.
        :type index: int
        :returns: A tuple of image and label.
        :rtype: typle[Any, int]
        """
        return self.data[index]
