import torch 

x = torch.tensor(2.0) #Input feature
y_true = torch.tensor(10.0) #Actual output
w = torch.tensor(3.0,requires_grad=True) #weights
b = torch.tensor(1.0,requires_grad=True) #bias
print("Initial Weight: ",w)
print("Initial Bias: ",b)
y_pred = w * x + b
loss = (y_pred - y_true)**2
print("Prediction:", y_pred)
print("Loss", loss)

#BackPropagation
loss.backward()
print("Gradient of w:",w.grad)
print("Gradient of b:",b.grad)

learning_rate = 0.01
with torch.no_grad():
    w = w - learning_rate * w.grad #W(new) = W(old) - LearningRate * w.grad
    b = b - learning_rate * b.grad

print("Updated w: ", w)
print("Updated b: ", b)
