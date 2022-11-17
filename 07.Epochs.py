import numpy as np

#Defining activation function
def step(x):
  return 1 if x > 0 else 0
  
#Training Data
data = [[0,0],
	  [0,1],
	  [1,0],
	  [1,1]]

#Target Values
y = [0, 1, 1, 1] 

#Initialising weights and bias
w = np.random.randn(2)
b = np.random.randn()
epochs = 30
alpha = 0.1

#loop over desired number of epochs
for i in range(epochs):
     #loop over individual data point
	for(x, target) in zip(data, y):
        #performing the dot product between the input data and the weights
		z = sum(x * w) + b
            #passing the net input through the activation function
		pred = step(z)
            #calculating the error
		error = target - pred
            #updating the weights and bias
		for index,value in enumerate(x):
			w[index] += alpha * error * value
			b += alpha * error

#Testing
test = [0, 1]
u = w[0] * test[0] + w[1] * test[1] + b
pred = step(u)
print(pred) 
