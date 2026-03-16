import torch

from tinyimagenet import TinyImageNet
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, Subset
from typing import Callable

TINYIMAGENET_TO_CIFAR10 = {
    "airplane": [],
    "automobile": [
        "beach wagon", "convertible", "limousine", "sports car"
    ],
    "bird": ["goose", "albatross"],
    "cat": ["tabby", "Persian cat", "Egyptian cat"],
    "deer": [],
    "dog": [
        "Chihuahua", "Yorkshire terrier", "golden retriever",
        "Labrador retriever", "German shepherd", "standard poodle"
    ],
    "frog": ["bullfrog", "tailed frog"],
    "horse": [],
    "ship": ["gondola", "lifeboat"],
    "truck": ["moving van", "police van"],
}

def load_tinyimagenet(
    root: str,
    train_val_partition: tuple[float, float],
    train_transform: Callable=ToTensor(),
    eval_transform: Callable=ToTensor(),
    verbose: bool=True
)-> tuple[Subset, Subset, Dataset]:
    """
    Load the TinyImageNet dataset directly as train, val, and test sets.
    Images are 64x64, apply a resize in transforms if using LeNet5.

    :param root: Directory to save cached files to.
    :type root: str
    :param train_val_partition: Percentages for how much of the train 
        data should remain train and how much should be for validation.
    :type train_val_partition: tuple[float, float]
    :param train_transform: The transformations that are added onto the
        train set.
    :type train_transform: Callable
    :param eval_transform: The transformations that are added onto the
        val and test sets.
    :type eval_transform: Callable
    :param verbose: Print info during process (DEFAULT=True)
    :type verbose: bool 
    :returns: Train, val, and test datasets, in that order.
    :rtype: tuple[Subset, Subset, Dataset]
    """
    assert sum(train_val_partition) == 1.0, "train_val_partition must sum to 1"

    training_data = TinyImageNet(
        root=root,
        split="train",
        transform=train_transform
    )
    evaluation_data = TinyImageNet(
        root=root,
        split="train",
        transform=eval_transform
    )
    test_dataset = TinyImageNet(
        root=root,
        split="val",
        transform=eval_transform
    )
    
    classes = [
        training_data.idx_to_words[i] for i in range(
            len(training_data.idx_to_words)
        )
    ]
    training_data.classes = classes
    evaluation_data.classes = classes
    test_dataset.classes = classes

    train_size = int(train_val_partition[0] * len(training_data))
    indices = torch.randperm(len(training_data))
    train_dataset = torch.utils.data.Subset(
        training_data,
        indices[:train_size]
    )
    val_dataset = torch.utils.data.Subset(
        evaluation_data,
        indices[train_size:]
    )
    if verbose:
        print(
            f"\033[30mDataset sizes:\n\tTrain: {len(train_dataset)} datapoints"
            f"\n\tVal:   {len(val_dataset)} datapoints"
            f"\n\tTest: {len(test_dataset)} datapoints\033[37m"
        )

    return train_dataset, val_dataset, test_dataset

def filter_subset(subset: Subset, index_remap: dict[int, int])-> Subset:
    # kept_indices = [
    #     i for i in subset.indices
    #     if subset.dataset.targets[i] in index_remap.keys()
    # ]
    # new_subset = Subset(subset.dataset, kept_indices)

    # for i in kept_indices:
    #     new_subset.dataset.targets[i] = index_remap[
    #         new_subset.dataset.targets[i]
    #     ]
    kept_indices = [
        i for i in subset.indices
        if subset.dataset.targets[i] != -1
    ]
    new_subset = Subset(subset.dataset, kept_indices)
    
    return new_subset

def filter_tinyimagenet_to_cifar10(
    train_dataset: Subset,
    val_dataset: Subset,
    test_dataset: Dataset
)-> tuple[Subset, Subset, Subset]:
    """
    Filter the TinyImageNet to only keep datapoints conforming with the
    classes present in cifar10.

    :param train_dataset: Train dataset.
    :type train_dataset: Subset
    :param val_dataset: Validation dataset.
    :type val_dataset: Subset
    :param test_dataset: Test dataset.
    :type test_dataset: Dataset
    """
    # print(len(test_dataset))
    # print(set(test_dataset.targets))
    tiny_classes = train_dataset.dataset.classes
    cifar10_classes = list(TINYIMAGENET_TO_CIFAR10.keys())
    
    # Create remapping table
    index_remap = {}
    for i, tiny_class in enumerate(tiny_classes):
        for j, cifar_class in enumerate(cifar10_classes):
            if any(
                keyword in tiny_class for keyword in \
                TINYIMAGENET_TO_CIFAR10[cifar_class]
            ):
                index_remap[i] = j
                break
    
    print(
        f"\033[30mRetaining {len(set(index_remap.values()))} classes from"
        f"{len(tiny_classes)} TinyImageNet classes, keeping {len(index_remap)}"
        f"TinyImageNet labels in dataset.\033[37m"
    )

    test_subset = Subset(test_dataset, list(range(len(test_dataset))))

    for ds in [
        train_dataset.dataset,
        val_dataset.dataset,
        test_subset.dataset
    ]:
        ds.targets = [
            index_remap[target] if target in index_remap.keys() else -1
            for target in ds.targets
        ]
        ds.classes = cifar10_classes

    filtered_train = filter_subset(train_dataset, index_remap)
    filtered_val = filter_subset(val_dataset, index_remap)
    filtered_test = filter_subset(test_subset, index_remap)

    # print(index_remap)
    # print(set(filtered_train.dataset.targets))
    # print(set(filtered_val.dataset.targets))
    # print(set(filtered_test.dataset.targets))

    return filtered_train, filtered_val, filtered_test
