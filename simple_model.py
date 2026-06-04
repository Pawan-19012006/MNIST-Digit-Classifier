import torch
import torch.nn as nn

class HousePredictionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer = nn.Linear(
            in_features = 3,
            out_features=1
        )
    def forward(self,x):
        output = self.layer(x)
        return output
    
#Creating a Model
model = HousePredictionModel()

#Loss function
criterion = nn.BCEWithLogitsLoss()

#Optimizer
optimizer = torch.optim.SGD(
    model.parameters(), #We feed the optimizer with the parameters ie, the bias and the weights 
    lr=0.01
)

# INPUT DATA
X = torch.tensor([
    [2.0, 8.0, 9.0],
    [9.0, 1.0, 2.0],
    [3.0, 7.0, 8.0],
    [8.0, 2.0, 1.0]
])

# TARGET LABELS
y = torch.tensor([
    [1.0],
    [0.0],
    [1.0],
    [0.0]
])

#Training Loop
for epoch in range(100):
    #Forward Pass
    logits = model(X)

    #Loss 
    loss = criterion(logits,y)

    #Clear the old grad
    optimizer.zero_grad()

    #BackProp
    loss.backward()

    #Update the params
    optimizer.step()

    print(f"Epoch {epoch} Loss: {loss.item()}")

#Test Prediction
test_input = torch.tensor([
    [2.0, 9.0, 9.0]
])

#Get Logits
logits = model(test_input)

#Convert to prob
probability = torch.sigmoid(logits)

print("Probability: ", probability)

#Final Prediction
prediction = (probability>=0.5).float()

print("Prediction:",prediction)