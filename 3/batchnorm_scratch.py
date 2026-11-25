"""
BatchNorm Implementation Exercise
Based on Andrej Karpathy's Neural Networks: Zero to Hero
https://github.com/karpathy/nn-zero-to-hero

This is a skeleton for you to implement BatchNorm from scratch.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# ============================================================================
# Part 1: Create a simple dataset
# ============================================================================

def create_dataset(n_samples=1000, input_dim=10, num_classes=2):
    """Create a simple synthetic classification dataset."""
    X = torch.randn(n_samples, input_dim)
    y = (X[:, 0] + X[:, 1] > 0).long()  # Simple decision boundary
    return TensorDataset(X, y), X, y


# ============================================================================
# Part 2: Implement BatchNorm from scratch
# ============================================================================

class BatchNorm1d:
    """
    Batch Normalization for 1D input (N, C) where N is batch size, C is features.

    During training:
    - Normalize using batch statistics (mean, variance)
    - Apply learnable affine transformation (scale and shift)
    - Track running statistics for inference

    During inference:
    - Normalize using running statistics (not batch statistics)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        Args:
            num_features: Number of features (C dimension)
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics update (typical: 0.1)
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = torch.ones(num_features)  # Scale parameter
        self.beta = torch.zeros(num_features)  # Shift parameter

        # Running statistics (for inference)
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

        # Gradients for learnable parameters
        self.dgamma = None
        self.dbeta = None

        # Cache for backward pass
        self.cache = {}

    def forward(self, x, training=True):
        """
        Forward pass of BatchNorm.

        Args:
            x: Input tensor of shape (N, C)
            training: Boolean, whether in training mode

        Returns:
            out: Normalized output of shape (N, C)

        TODO: Implement forward pass
        """

        mean_batch = x.mean(0)
        variance_batch = x.std(0)

        self.running_mean *= (1 - self.momentum)
        self.running_mean += mean_batch * self.momentum

        self.running_var *= (1 - self.momentum)
        self.running_var += variance_batch * self.momentum

        x_hat = (x - mean_batch)/(variance_batch**2 + self.eps)**0.5 #mistake: forgot to square the var-batch

        self.cache = {
            'mean'  :   mean_batch,
            'var'   :   variance_batch,
            'x_hat' :   x_hat,
            'x'     :   x,
            'N'     :   x.shape[0]
        }
        # print(f"DEBUG| x: {x.shape}, {self.gamma.shape}, {self.beta.shape}, {x_hat.shape}")
        # print(f"x_hat mean: {x_hat.mean():.4f}, std: {x_hat.std():.4f}")
        return self.gamma * x_hat + self.beta

    def backward(self, dout):
        """
        Backward pass of BatchNorm.

        Args:
            dout: Gradient of loss w.r.t. output, shape (N, C)

        Returns:
            dx: Gradient of loss w.r.t. input, shape (N, C)

        TODO: Implement backward pass

        Hints:
        - You'll need to use the cache stored in forward()
        - Compute gradients for gamma and beta
        - Compute gradient w.r.t. input
        """

        mean        =   self.cache["mean"]
        var         =   self.cache["var"]
        x_hat       =   self.cache["x_hat"]
        x           =   self.cache["x"]
        N           =   self.cache["N"]
        
        dx_hat = dout * self.gamma
        dvar = -(x - mean).sum(0)*(var + self.eps)**(-3/2)/2
        dmean = x.sum(0)/-(var**2 + self.eps)**0.5 + dvar*((x-mean).sum(0)*(-2/N)) # try optimizing this
        din = dx_hat/((var**2 + self.eps)**0.5) + dvar*((x - mean).sum(0)*(2/N)) + dmean/N
        self.dgamma = (dout * x_hat).sum(0)
        self.dbeta = dout #mistake: accidentaly have beta instead of dbeta

        return din


# ============================================================================
# Part 3: Simple Neural Network with BatchNorm
# ============================================================================

class SimpleNet:
    """
    Simple 3-layer network: Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm -> ReLU -> Linear
    """

    def __init__(self, input_dim=10, hidden_dim=50, num_classes=2):
        self.layers = []

        # Layer 1: Linear
        self.W1 = torch.randn(input_dim, hidden_dim) * 0.01
        self.b1 = torch.zeros(hidden_dim)
        self.bn1 = BatchNorm1d(hidden_dim)

        # Layer 2: Linear
        self.W2 = torch.randn(hidden_dim, hidden_dim) * 0.01
        self.b2 = torch.zeros(hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)

        # Layer 3: Linear (output)
        self.W3 = torch.randn(hidden_dim, num_classes) * 0.01
        self.b3 = torch.zeros(num_classes)

        self.cache = {}

    def forward(self, x, training=True):
        """
        Forward pass through the network.

        TODO: Implement forward pass
        - Apply layers in sequence
        - Use BatchNorm between linear layers and activations
        - Use ReLU activations
        """
        relu = nn.ReLU()

        # print("DEBUG: Linear+ReLU 1")

        p1 = x @ self.W1 + self.b1
        y1 = self.bn1.forward(p1)
        a1 = relu(y1)

        # print("DEBUG: BN1")

        # x_hat1 = self.bn1.forward(a1) #mistake: I accendaly have b1 here instead

        # print("DEBUG: Linear+ReLU 2")
        
        p2 = a1 @ self.W2 + self.b2
        y2 = self.bn2.forward(p2)
        a2 = relu(y2)

        # x_hat2 = self.bn2.forward(a2)

        # print("DEBUG: Linear+ReLU 3")

        y3 = a2 @ self.W3 + self.b3

        self.cache = {
            "x"         :   x,
            "mask1"     :   (y1 > 0),
            "x1"        :   a1,
            "mask2"     :   (y2 > 0),
            "x2"        :   a2
        }

        return torch.sigmoid(y3)
    
    def backward(self, dout):
        """
        Backward pass through the network.

        TODO: Implement backward pass
        - Backprop through all layers
        - Update gradients for weights and biases
        """     
        x = self.cache['x']
        mask1 = self.cache['mask1']
        x1 = self.cache['x1']
        mask2 = self.cache['mask2']
        x2 = self.cache['x2']

        dy3 = dout  
        self.W3.grad = x2.T @ dy3 
        self.b3.grad = dy3.sum(0)
        dx_hat2 = dy3 @ self.W3.T

        da2 = self.bn2.backward(dx_hat2)

        dy2 = da2 * mask2
        self.W2.grad = x1.T @ dy2
        self.b2.grad = dy2.sum(0)
        dx_hat1 = dy2 @ self.W2.T

        da1 = self.bn1.backward(dx_hat1)

        dy1 = da1 * mask1
        self.W1.grad = x.T @ dy1
        self.b1.grad = dy1.sum(0)

        pass

    def parameters(self):
        """Return all learnable parameters (for optimization)."""
        params = [
            self.W1, self.b1, self.bn1.gamma, self.bn1.beta,
            self.W2, self.b2, self.bn2.gamma, self.bn2.beta,
            self.W3, self.b3
        ]
        return params

    def zero_grad(self):
        """Zero out all gradients."""
        for p in self.parameters():
            p.grad = None


# ============================================================================
# Part 4: Training loop and tests
# ============================================================================

def train_epoch(model, train_loader, lr=0.01):
    """Train for one epoch."""
    total_loss = 0
    for X_batch, y_batch in train_loader:
        # Forward pass
        logits = model.forward(X_batch, training=True)

        # Simple cross-entropy loss (you can use your own implementation)
        # For now, we'll just compute a simple mean squared error as placeholder
        loss = ((logits - torch.nn.functional.one_hot(y_batch, 2).float()) ** 2).mean()

        # Backward pass
        dout = 2 * (logits - torch.nn.functional.one_hot(y_batch, 2).float()) / len(y_batch)
        model.backward(dout)

        # Update parameters
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

        model.zero_grad()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, X, y):
    """Evaluate accuracy on a dataset."""
    logits = model.forward(X, training=False)
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == y).float().mean().item()
    return accuracy


# ============================================================================
# Part 5: Main script - Run this to test your implementation
# ============================================================================

if __name__ == "__main__":
    print("BatchNorm Implementation Exercise")
    print("=" * 50)

    # Create dataset
    train_dataset, X_train, y_train = create_dataset(n_samples=1000, input_dim=10, num_classes=2)
    test_dataset, X_test, y_test = create_dataset(n_samples=200, input_dim=10, num_classes=2)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = SimpleNet(input_dim=10, hidden_dim=50, num_classes=2)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print()

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader, lr=0.01)
        train_acc = evaluate(model, X_train, y_train)
        test_acc = evaluate(model, X_test, y_test)

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    print()
    print("=" * 50)
    print("Training complete!")
    print("Check if:")
    print("✓ Training loss decreases over time")
    print("✓ Training accuracy improves")
    print("✓ BatchNorm properly normalizes activations")
    print("✓ Running statistics are updated during training")
