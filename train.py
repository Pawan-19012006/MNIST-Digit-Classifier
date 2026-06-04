import torch 

print("Pytorch Version: ", torch.__version__)
print("MPS Available: ", torch.backends.mps.is_available())
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using Device: ",device)
