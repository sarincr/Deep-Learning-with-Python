import numpy as np

X = np.array([ [ 0, 0],
               [ 0, 1 ],
               [ 1, 0 ],
               [ 1, 1 ] ])
print(X)

y = np.array([[0,0,0,1]]).T
print(y)

n_inputs = 2
n_outputs = 1

Wo = [[1],[1]]
print(Wo)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def SigmoidBack(z):

    return z * (1 - z)

for n in range(10000):
    # forward propagation
    l1 = sigmoid(np.dot(X, Wo)+1)
    
    # compute the loss
    l1_error = y - l1
    #print("l1_error:\n", l1_error)
    
    # multiply the loss by the slope of the sigmoid at l1
    l1_delta = l1_error * SigmoidBack(l1)
    Wo += np.dot(X.T, l1_delta)


print(Wo)
 
