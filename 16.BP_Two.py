import numpy as np
from matplotlib import pyplot as plt

def sigmoid(x, derive=False):
    if derive:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# Define the data set XOR
X = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
])

y = np.array([[0],
              [1],
              [1],
              [0]
             ])

# Define a learning rate
eta = 3
# Define the number of epochs for learning
epochs = 20000

# Initialize the weights with random numbers
w01 = np.random.random((len(X[0]), 5))
w12 = np.random.random((5, 1))

# Start feeding forward and backpropagate *epochs* times.
for epoch in range(epochs):
    # Feed forward
    z_h = np.dot(X, w01)
    a_h = sigmoid(z_h)

    z_o = np.dot(a_h, w12)
    a_o = sigmoid(z_o)

    # Calculate the error
    a_o_error = ((1 / 2) * (np.power((a_o - y), 2)))

    # Backpropagation
    ## Output layer
    delta_a_o_error = a_o - y
    delta_z_o = sigmoid(a_o,derive=True)
    delta_w12 = a_h
    delta_output_layer = np.dot(delta_w12.T,(delta_a_o_error * delta_z_o))

    ## Hidden layer
    delta_a_h = np.dot(delta_a_o_error * delta_z_o, w12.T)
    delta_z_h = sigmoid(a_h,derive=True)
    delta_w01 = X
    delta_hidden_layer = np.dot(delta_w01.T, delta_a_h * delta_z_h)

    w01 = w01 - eta * delta_hidden_layer
    w12 = w12 - eta * delta_output_layer
print(a_o,y)
