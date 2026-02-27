import numpy as np

# ── Seed for reproducibility ──────────────────────────────────────
np.random.seed(42)

# ── Toy dataset (XOR problem) ─────────────────────────────────────
X = np.array([[0,0],[0,1],[1,0],[1,1]])  # shape (4, 2)
y = np.array([[0],[1],[1],[0]])           # shape (4, 1)

# ── Initialize weights & biases randomly ─────────────────────────
W1 = np.random.randn(2, 3) * 0.1   # input→hidden   (2 inputs, 3 neurons)
b1 = np.zeros((1, 3))
W2 = np.random.randn(3, 1) * 0.1   # hidden→output  (3 neurons, 1 output)
b2 = np.zeros((1, 1))

learning_rate = 0.1

# ── Activation function ───────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# ── Loss function (Binary Cross-Entropy) ─────────────────────────
def compute_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

# ── Forward pass ─────────────────────────────────────────────────
def forward(X):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

# ── Backward pass ────────────────────────────────────────────────
def backward(X, y, Z1, A1, Z2, A2):
    m = X.shape[0]  # number of samples

    dL_dA2 = -(y / (A2 + 1e-8)) + (1 - y) / (1 - A2 + 1e-8)
    dL_dZ2 = dL_dA2 * sigmoid_derivative(Z2)
    dL_dW2 = np.dot(A1.T, dL_dZ2) / m
    db2 = np.mean(dL_dZ2, axis=0, keepdims=True)

    dL_dA1 = np.dot(dL_dZ2, W2.T)
    dL_dZ1 = dL_dA1 * sigmoid_derivative(Z1)
    dL_dW1 = np.dot(X.T, dL_dZ1) / m
    db1 = np.mean(dL_dZ1, axis=0, keepdims=True)

    return dL_dW1, db1, dL_dW2, db2

# ── Weight update (gradient descent) ─────────────────────────────
def update_weights(dW1, db1, dW2, db2):
    global W1, b1, W2, b2
  
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2


# ── Training loop ─────────────────────────────────────────────────
for epoch in range(10000):
    Z1, A1, Z2, A2 = forward(X)
    loss = compute_loss(y, A2)
    dW1, db1, dW2, db2 = backward(X, y, Z1, A1, Z2, A2)
    update_weights(dW1, db1, dW2, db2)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

# ── Final predictions ─────────────────────────────────────────────
_, _, _, A2_final = forward(X)
print("\nFinal predictions:")
print(np.round(A2_final, 3))
print("Expected: [[0],[1],[1],[0]]")