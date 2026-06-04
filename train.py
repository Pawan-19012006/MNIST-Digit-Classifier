import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn

train_dataset = datasets.MNIST(
    root = "data",
    train = True, #splits into train and test
    download = True,
    transform = transforms.ToTensor() #raw mnist images are in the format of PIL, need to convert to Tensors
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32, #Batching of 32 images together
    shuffle=True #the order of the images will shuffle, else the labels will be stagnant and it creates learning bias
)

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784,128) #Layer 1 is the first hidden layer, it has 784 input features and 128 Neurons
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128,10)#Layer 2 is the output layer, that only posesses 10 neurons, becoz each neuron represents a digit

    def forward(self,x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
    
model = MNISTModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

for epoch in range(5):
    running_loss=0
    for images,labels in train_loader:
        images = images.view(images.shape[0],-1)
        outputs = model(images)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss = running_loss + loss.item()
    epoch_loss = running_loss/len(train_loader)
   # print(f"Epoch {epoch+1}, Loss: {epoch_loss}")

model.eval() #This will state that the model is in inference/evaluation mode

image,label = train_dataset[0]
image = image.view(1,784)
with torch.no_grad():
    outputs = model(image)
    prediction = torch.argmax(outputs, dim=1)
print(f"Predicted:{prediction.item()}")
print(f"Actual: {label}")
          
