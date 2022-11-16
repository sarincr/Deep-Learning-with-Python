import math

X = [0.7, 0.1, 0.2]
Y = [1, 0, 0]

loss = -(math.log(X[0]) * Y[0] +
         math.log(X[1]) * Y[1] +
         math.log(X[2]) * Y[2])

print(loss)

