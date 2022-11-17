
import pandas as pd
import numpy as np

def threshold(x):
    return 1 if x >= 2 else 0

def fire(data, weights, output):
    for x in data:
        weighted_sum = np.inner(x, weights)
        output.append(threshold(weighted_sum))

data = [[0,0], [0,1], [1,0], [1,1]]

weights = [1, 1]
output = []

fire(data, weights, output)

t = pd.DataFrame(index=None)
t['X1'] = [0, 0, 1, 1]
t['x2'] = [0, 1, 0, 1]
t['y'] = pd.Series(output)

print(t)
