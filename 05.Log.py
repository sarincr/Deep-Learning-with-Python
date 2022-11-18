import math
import numpy as np

X = [.1, -.010, .010]
Y = [.11, .1, .10]

K = (X[0] * Y[0] +   X[1] * Y[1] +    X[2] * Y[2])



print(K)


a=1/(1 + np.exp(-K))

print(a)
