import math
import numpy as np

X = [1000, 20000, 30000]
Y = [11, 4, 10]

loss = (X[0] * Y[0] +   X[1] * Y[1] +    X[2] * Y[2])

z=math.log(loss)

print(z,loss)


a=1/(1 + np.exp(-loss))

print(a)
