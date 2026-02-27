import numpy as np
import matplotlib.pyplot as plt

def run_experiment(lr, weight_scale, epochs=10000):
    np.random.seed(42)

    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])

    W1 = np.random.randn(2, 3) * weight_scale
    b1 = np.zeros((1, 3))
    W2 = np.random.randn(3, 1) * weight_scale
    b2 = np.zeros((1, 1))

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(z):
        return sigmoid(z) * (1 - sigmoid(z))

    def compute_loss(y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred + 1e-8) + (1-y_true) * np.log(1-y_pred + 1e-8))

    def forward(X):
        Z1 = np.dot(X, W1) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = sigmoid(Z2)
        return Z1, A1, Z2, A2

    def backward(X, y, Z1, A1, Z2, A2):
        m = X.shape[0]
        dL_dA2 = -(y/(A2+1e-8)) + (1-y)/(1-A2+1e-8)
        dL_dZ2 = dL_dA2 * sigmoid_derivative(Z2)
        dL_dW2 = np.dot(A1.T, dL_dZ2) / m
        db2    = np.mean(dL_dZ2, axis=0, keepdims=True)
        dL_dA1 = np.dot(dL_dZ2, W2.T)
        dL_dZ1 = dL_dA1 * sigmoid_derivative(Z1)
        dL_dW1 = np.dot(X.T, dL_dZ1) / m
        db1    = np.mean(dL_dZ1, axis=0, keepdims=True)
        return dL_dW1, db1, dL_dW2, db2

    loss_history = []
    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward(X)
        loss = compute_loss(y, A2)
        loss_history.append(loss)
        dW1, db1, dW2, db2 = backward(X, y, Z1, A1, Z2, A2)
        W1_ = W1 - lr * dW1
        b1_ = b1 - lr * db1
        W2_ = W2 - lr * dW2
        b2_ = b2 - lr * db2
        W1[:], b1[:], W2[:], b2[:] = W1_, b1_, W2_, b2_

    return loss_history

# Run these 4 experiments and plot them all on one chart
# Experiment 1: lr=0.1,  weight_scale=0.1  (too slow)
# Experiment 2: lr=1.1,  weight_scale=0.1  (our current - baseline)
# Experiment 3: lr=5.0,  weight_scale=0.1  (too high - watch it diverge)
# Experiment 4: lr=1.1,  weight_scale=0.5  (better init - does the cliff disappear?)

experiments = [
    (0.1, 0.1, "lr=0.1, ws=0.1 (too slow)"),
    (1.1, 0.1, "lr=1.1, ws=0.1 (baseline)"),
    (5.0, 0.1, "lr=5.0, ws=0.1 (diverge)"),
    (1.1, 0.5, "lr=1.1, ws=0.5 (better init)"),
]

plt.figure(figsize=(12, 6))
for lr, ws, label in experiments:
    loss_history = run_experiment(lr, ws)
    plt.plot(loss_history, label=label)

plt.title('Loss Curves: LR & Weight Init Experiments')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')   # log scale makes differences more visible
plt.grid(True)
plt.show()