import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Initialize remaining weight matrices and biases
W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 4) * 0.5
b2 = np.zeros((1, 4))
W3 = np.random.randn(4, 1) * 0.5
b3 = np.zeros((1, 1))

learning_rate = 0.1

# ── Activations ───────────────────────────────────────────────────
# ReLU — hard zero for negatives, neurons can die
def relu(z): 
    return np.maximum(0, z)

# Leaky ReLU — small slope for negatives, neurons stay alive
def leaky_relu(z, alpha=0.01): 
    return np.where(z > 0, z, 0.01 * z)

def leaky_relu_derivative(z, alpha=0.01):
    # return 1 where z > 0, else alpha
    return np.where(z > 0, 1, alpha)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def compute_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-8) + (1-y_true) * np.log(1-y_pred + 1e-8))

# ── Forward pass ──────────────────────────────────────────────────
def forward(X):
    Z1 = np.dot(X, W1) + b1;  
    A1 = leaky_relu(Z1)
    Z2 = np.dot(A1, W2) + b2; 
    A2 = leaky_relu(Z2)
    Z3 = np.dot(A2, W3) + b3; 
    A3 = sigmoid(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# ── Backward pass ─────────────────────────────────────────────────
def backward(X, y, Z1, A1, Z2, A2, Z3, A3):
    m = X.shape[0]
    dA3 = -(y/(A3+1e-8)) + (1-y)/(1-A3+1e-8)
    dZ3 = dA3 * sigmoid_derivative(Z3)
    dW3 = np.dot(A2.T, dZ3) / m
    db3 = np.mean(dZ3, axis=0, keepdims=True)

    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * leaky_relu_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * leaky_relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

    

# ── Weight update ─────────────────────────────────────────────────
def update_weights(dW1, db1, dW2, db2, dW3, db3):
    global W1, b1, W2, b2, W3, b3
    # update all 3 layers
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

# ── Training loop ─────────────────────────────────────────────────
loss_history = []

for epoch in range(10000):
    Z1, A1, Z2, A2, Z3, A3 = forward(X)
    loss = compute_loss(y, A3)
    loss_history.append(loss)
    dW1, db1, dW2, db2, dW3, db3 = backward(X, y, Z1, A1, Z2, A2, Z3, A3)
    update_weights(dW1, db1, dW2, db2, dW3, db3)

    if (epoch+1) % 1000 == 0:
        print(f"Epoch {epoch+1:5d} | Loss: {loss:.4f} | "
              f"Grad W1: {np.mean(np.abs(dW1)):.6f} | "
              f"Grad W2: {np.mean(np.abs(dW2)):.6f} | "
              f"Grad W3: {np.mean(np.abs(dW3)):.6f}")

# ── Plot ──────────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(loss_history, color='steelblue', linewidth=1.5)
plt.title('3-Layer Network with LeakyReLU — Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True, alpha=0.4)
plt.show()

# ── Predictions ───────────────────────────────────────────────────
_, _, _, _, _, A3_final = forward(X)
print("\nFinal predictions:")
print(np.round(A3_final, 3))
print("Expected: [[0],[1],[1],[0]]")