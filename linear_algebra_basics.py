import torch 

x = torch.tensor([2.0,3.0,4.0])
w = torch.tensor([0.5,0.2,0.8])
b = torch.tensor(1.0)
y = torch.dot(w,x) + b # y = w1x1 + w2x2 + w3x3 + b
print(y)