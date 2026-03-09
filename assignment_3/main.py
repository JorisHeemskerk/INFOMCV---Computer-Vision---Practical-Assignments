from torchvision import datasets

from data import load_datasets


def main()-> None:
    train, val, test = load_datasets(datasets.CIFAR10, "assignment_3/data/", (.8, .2))

if __name__ == "__main__":
    main()
