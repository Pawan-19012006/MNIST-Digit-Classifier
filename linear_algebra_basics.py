import torch 
import torch.nn as nn

#We introduce this model in order to perform the forward pass
layer = nn.Linear(
    in_features=3,
    out_features=2
)

x = torch.tensor([2.0,3.0,4.0]) # providing the input features, 3 ip features, as specified
output = layer(x) # applying the model
print(output) # getting the output of the given inputs, ie, 2 outputs as specified
print(layer.weight)
print(layer.bias)