## IMPORTS
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn

## Dataset Creation
train_dataset = datasets.MNIST(
    root = "data",
    train = True, #splits into train and test
    download = True,
    transform = transforms.ToTensor() #raw mnist images are in the format of PIL, need to convert to Tensors
)

test_dataset = datasets.MNIST(
    root = "data",
    train = False,
    download=True,
    transform = transforms.ToTensor()
)

# DataLoaders Creation
train_loader = DataLoader( #We are splitting the train data into batches each containing 32 images
    dataset=train_dataset,
    batch_size=32, #Batching of 32 images together
    shuffle=True #the order of the images will shuffle, else the labels will be stagnant and it creates learning bias
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size = 32,
    shuffle=False
)

## Blueprint of the model
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__() #input layer layer only passes the features, there will be no activation taking place in there, also initially the number of neurons will be equal to the number of features in the input layer, but after that it will keep reducing
        self.layer1 = nn.Linear(784,128) #Layer 1 is the first hidden layer, it has 784 input features and 128 Neurons
        self.relu = nn.ReLU() #it is the activation function that activates the neurons and also introduces a nonlinearity
        self.layer2 = nn.Linear(128,10)#Layer 2 is the output layer, that only posesses 10 neurons, becoz each neuron represents a digit

    def forward(self,x): #forward function is responsible for the pathway/flow of the layers, what layer to be passed after another will be defined in here
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

## Object Creation
model = MNISTModel() #We use the mnistmodel which is the class we created above
criterion = nn.CrossEntropyLoss() #it is the one used for the loss function finding
optimizer = torch.optim.SGD( #We use the Stochastic Gradient Descent Optimizer in here
    model.parameters(), #We feed all the parameters to it, ie the weights and the bias
    lr=0.01 #Learning rate is constant and stable throughout the optimization in case of SGD
)

train_losses = []
test_accuracies = []

## Training Loop Begins
for epoch in range(5): #One epoch will contain all the forward pass, backward pass and optimization of the whole train dataset, example if there are 60000 images in dataset, we split it into 32 batches, so there will be 1875 batches alltogether, so one epoch will contain 1875 forward pass, 1875 backprop and 1875 optimizer updates
    running_loss=0 #initiating the running loss variable
    for images,labels in train_loader: #we fetch the images and labels from the train loader(train loader sends each batch data)
        images = images.view(images.shape[0],-1) # Here we flatten(.view can reshape the images) the images coz, initially it will contain [batchsize,h,w,colorcode], now after flattening it will turn to [batchsize,features] so that it can be passed as the feature vector
        outputs = model(images) #here the init function and forward function will be called automatically on the inside
        loss = criterion(outputs,labels)#Finding the loss function by using the output labels we found above and actual labels
        optimizer.zero_grad() #clearing the optimizer so that it does not accumulate the gradients
        loss.backward() #backpropagation starts, here the backward function is automatically derived by pytorch using Autograd which creates the computational graph
        optimizer.step() #.step is the one that actually updates the parameters, here is where new weights and biases are found using the formula and then are updated
        running_loss = running_loss + loss.item() #Running loss adds the loss from each batch(loss.item gives the loss number from each batch)
    epoch_loss = running_loss/len(train_loader) #gives the average loss for the whole epoch once the for loop gets executed, here we get runningloss/1875 since there are 1875 batches in the trainloader
    train_losses.append(epoch_loss)


    model.eval() #This will state that the model is in evaluation mode, has to be specified before the testing and evaluation
    correct = 0
    total = 0
    with torch.no_grad():#it states not to track and keep history of gradients
        for images, labels in test_loader:
            images = images.view(images.shape[0],-1) #flattening
            outputs = model(images) #predicting
            predictions = torch.argmax(outputs,dim=1) #when there are 32 images, each image will get an output value, we take the highest value out of it 
            correct += (predictions==labels).sum().item()
            total+=labels.size(0)
    accuracy = 100 * correct/total
    test_accuracies.append(accuracy)
    print(f"Accuracy:{accuracy:.2f}%")
    model.train() #This will change the state of the model back to training mode from eval mode

## Loss Curve
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

