from torchvision import datasets

from data import load_datasets
from lenet import Lenet5


def main()-> None:
    # train, val, test = load_datasets(datasets.CIFAR10, "assignment_3/data/", (.8, .2))

    model = Lenet5()
    print(model)


if __name__ == "__main__":
    main()
