import torch
from torchvision import datasets
from torchvision import transforms

train_dataset = datasets.MNIST(
    root = "data",
    train = True, #splits into train and test
    download = True,
    transform = transforms.ToTensor() #raw mnist images are in the format of PIL, need to convert to Tensors
)

print(len(train_dataset))