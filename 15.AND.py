import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

w = np.array([0.2, 0.3, 0.8])
b = 0.5
x = np.array([0.5, 0.6, 0.1])

print(w)

print(b)

z = w.dot(x) + b
print("z:", z)
print("a:", sigmoid(z))

def activation(z):
    if z > 0:
        return 1
    return 0

w = np.array([1, 1])
b = -1

print(w)

print(b)

x = np.array([0, 0])
print("0 AND 0:", activation(w.dot(x) + b))

x = np.array([0, 0])
print("0 AND 0:", activation(w.dot(x) + b))
x = np.array([1, 0])
print("1 AND 0:", activation(w.dot(x) + b))
x = np.array([0, 1])
print("0 AND 1:", activation(w.dot(x) + b))
x = np.array([1, 1])
print("1 AND 1:", activation(w.dot(x) + b))
