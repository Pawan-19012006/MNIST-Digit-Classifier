import torch 

x = torch.tensor(2.0) #Input feature
w = torch.tensor(3.0,requires_grad=True) #weights
b = torch.tensor(1.0,requires_grad=True) #bias
y = w * x + b
print(y)
y.backward()
print(w.grad)
print(b.grad)