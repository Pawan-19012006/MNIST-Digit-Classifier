# MNIST Digit Classifier using PyTorch

This project implements a **Fully Connected Neural Network (Dense Neural Network)** to classify handwritten digits from the MNIST dataset.

The goal of this project was not just to make the model work, but to deeply understand:

- tensors
- autograd
- forward propagation
- backpropagation
- batching
- neural network architecture
- logits and loss functions
- training and inference pipelines

---

# Project Overview

The model classifies grayscale handwritten digits (`0-9`) from the MNIST dataset.

Each image:

- size: `28 x 28`
- grayscale (`1 channel`)
- flattened into a `784-dimensional vector`

The architecture used:

```text
784 → 128 → 10
```

Where:

- `784` = flattened image pixels
- `128` = hidden neurons
- `10` = output classes (digits `0-9`)

---

# Technologies Used

- Python
- PyTorch
- Torchvision
- Matplotlib

---

# Concepts Implemented

## PyTorch Fundamentals

- Tensor operations
- Tensor reshaping
- Tensor flattening
- Gradient computation
- Computational graph
- Autograd engine

---

## Neural Network Concepts

- `nn.Module`
- `nn.Linear`
- Hidden layers
- ReLU activation
- Logits
- Forward propagation
- Backpropagation
- Gradient descent
- SGD optimizer

---

## Data Pipeline

- `torchvision.datasets.MNIST`
- `DataLoader`
- Mini-batching
- Dataset iteration
- Batch processing

---

# Training Pipeline

Implemented:

- Forward pass
- Loss computation
- Gradient clearing
- Backward propagation
- Weight updates

Training Flow:

```text
Images
   ↓
Flatten
   ↓
Model
   ↓
Logits
   ↓
CrossEntropyLoss
   ↓
Backward Pass
   ↓
Optimizer Step
```

---

# Model Architecture

```python
class MNISTModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.layer1 = nn.Linear(784, 128)

        self.relu = nn.ReLU()

        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.layer1(x)

        x = self.relu(x)

        x = self.layer2(x)

        return x
```

---

# Loss Function

Used:

```python
nn.CrossEntropyLoss()
```

Why?

- Suitable for multiclass classification
- Internally applies Softmax
- Expects raw logits from model

---

# Optimizer

Used:

```python
torch.optim.SGD()
```

Concept learned:

```text
Weights = Weights - LearningRate × Gradient
```

---

# Dataset Details

Dataset:

- MNIST Handwritten Digits Dataset

Training Images:

- 60,000

Testing Images:

- 10,000

---

# Tensor Shapes Learned

Single Image:

```text
[1, 28, 28]
```

Batch of Images:

```text
[32, 1, 28, 28]
```

Flattened Batch:

```text
[32, 784]
```

Model Output:

```text
[32, 10]
```

---

# Accuracy Achieved

Final Test Accuracy:

```text
~93%
```

using a simple fully connected neural network.

---

# Key Learnings

This project helped in deeply understanding:

- how tensors flow through neural networks
- why flattening is needed
- how batching works
- why ReLU introduces nonlinearity
- how logits are converted into predictions
- how gradients update weights
- how neural networks learn representations

---

# Future Improvements

Planned next steps:

- CNN-based MNIST classifier
- Model saving/loading
- Softmax probability visualization
- Better architecture experimentation
- GPU acceleration

---

# Run The Project

Install dependencies:

```bash
pip install torch torchvision matplotlib
```

Run training:

```bash
python train.py
```

---

# Author

Pawan Eswaran
