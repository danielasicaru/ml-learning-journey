import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# ── Dataset — two interleaved half-circles (non-linear, like XOR but bigger) ──
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
y = y.reshape(-1, 1)
# Normalize the data
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Hint: use the first 160 samples for train, last 40 for val
X_train, X_val = X[:160], X[160:]
y_train, y_val = y[:160], y[160:]

# ── Weights ───────────────────────────────────────────────────────
np.random.seed(42)
W1 = np.random.randn(2, 8) * 0.5
b1 = np.zeros((1, 8))
W2 = np.random.randn(8, 8) * 0.5
b2 = np.zeros((1, 8))
W3 = np.random.randn(8, 1) * 0.5
b3 = np.zeros((1, 1))

learning_rate = 0.5

def leaky_relu(z):       return np.where(z > 0, z, 0.01 * z)
def leaky_relu_deriv(z): return np.where(z > 0, 1, 0.01)
def sigmoid(z):          return 1 / (1 + np.exp(-z))
def sigmoid_deriv(z):    return sigmoid(z) * (1 - sigmoid(z))
def loss_fn(yt, yp):     return -np.mean(yt*np.log(yp+1e-8)+(1-yt)*np.log(1-yp+1e-8))

def forward(X):
    Z1=np.dot(X,W1)+b1;  A1=leaky_relu(Z1)
    Z2=np.dot(A1,W2)+b2; A2=leaky_relu(Z2)
    Z3=np.dot(A2,W3)+b3; A3=sigmoid(Z3)
    return Z1,A1,Z2,A2,Z3,A3

def backward(X,y,Z1,A1,Z2,A2,Z3,A3):
    m=X.shape[0]
    dA3=-(y/(A3+1e-8))+(1-y)/(1-A3+1e-8)
    dZ3=dA3*sigmoid_deriv(Z3)
    dW3=np.dot(A2.T,dZ3)/m; db3=np.mean(dZ3,axis=0,keepdims=True)
    dA2=np.dot(dZ3,W3.T)
    dZ2=dA2*leaky_relu_deriv(Z2)
    dW2=np.dot(A1.T,dZ2)/m; db2=np.mean(dZ2,axis=0,keepdims=True)
    dA1=np.dot(dZ2,W2.T)
    dZ1=dA1*leaky_relu_deriv(Z1)
    dW1=np.dot(X.T,dZ1)/m; db1=np.mean(dZ1,axis=0,keepdims=True)
    return dW1,db1,dW2,db2,dW3,db3

def update_weights(dW1,db1,dW2,db2,dW3,db3):
    global W1,b1,W2,b2,W3,b3
    W1-=learning_rate*dW1; b1-=learning_rate*db1
    W2-=learning_rate*dW2; b2-=learning_rate*db2
    W3-=learning_rate*dW3; b3-=learning_rate*db3

# ── Training loop with early stopping ────────────────────────────
train_losses, val_losses = [], []

# implement early stopping variables
patience = 200
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(5000):
    # Train
    Z1,A1,Z2,A2,Z3,A3 = forward(X_train)
    train_loss = loss_fn(y_train, A3)
    dW1,db1,dW2,db2,dW3,db3 = backward(X_train,y_train,Z1,A1,Z2,A2,Z3,A3)
    update_weights(dW1,db1,dW2,db2,dW3,db3)
    train_losses.append(train_loss)

    # Compute validation loss (forward pass on X_val only, no backward)
    Z1_val,A1_val,Z2_val,A2_val,Z3_val,A3_val = forward(X_val)
    val_loss = loss_fn(y_val, A3_val)
    val_losses.append(val_loss)

    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

    if (epoch+1) % 500 == 0:
        print(f"Epoch {epoch+1:4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# ── Plot both curves ──────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Training and Validation Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.4)
plt.show()