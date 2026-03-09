from torchvision import datasets

from data import load_datasets, to_dataloaders


def main()-> None:
    train, val, test =load_data(datasets.CIFAR10, "assignment_3/data/", (.8, .2))

if __name__ == "__main__":
    main()
