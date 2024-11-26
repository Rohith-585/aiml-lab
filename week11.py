import numpy as np

# Input (Hours Studied, Hours Slept) and Output (Test Scores normalized to 0-1)
X = np.array([[2, 9], [1, 5], [3, 6]])
y = np.array([[92], [86], [89]]) / 100

# Activation Function and its Derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Hyperparameters and Network Configuration
epochs = 10000  # Number of training iterations
lr = 0.1  # Learning rate
input_neurons = 2  # Features in the dataset
hidden_neurons = 3  # Neurons in the hidden layer
output_neurons = 1  # Neurons in the output layer

# Initialize weights and biases
wh = np.random.uniform(size=(input_neurons, hidden_neurons))  # Input to hidden weights
bh = np.random.uniform(size=(1, hidden_neurons))  # Hidden layer bias
wo = np.random.uniform(size=(hidden_neurons, output_neurons))  # Hidden to output weights
bo = np.random.uniform(size=(1, output_neurons))  # Output layer bias

# Training the Network
for _ in range(epochs):
    # Forward Propagation
    hidden_input = np.dot(X, wh) + bh
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, wo) + bo
    predicted_output = sigmoid(final_input)

    # Backpropagation
    output_error = y - predicted_output
    output_delta = output_error * sigmoid_derivative(predicted_output)

    hidden_error = output_delta.dot(wo.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    # Update Weights and Biases
    wo += hidden_output.T.dot(output_delta) * lr
    bo += np.sum(output_delta, axis=0, keepdims=True) * lr
    wh += X.T.dot(hidden_delta) * lr
    bh += np.sum(hidden_delta, axis=0, keepdims=True) * lr

# Display Results
print("Input:\n", X)
print("Actual Output:\n", y)
print("Predicted Output:\n", predicted_output)
