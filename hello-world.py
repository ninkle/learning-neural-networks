# the "hello world" of neural networks
# http://iamtrask.github.io/2015/07/12/basic-python-network/
import numpy as np


# define our sigmoid function
# sigmoid takes some number x and converts it to a binary value between 0 and 1
# we also have the option of computing the derivative of a sigmoid
def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input dataset
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

print("input dataset:")
print(X)

# initialize output dataset with the transpose function T
# after the transpose, y is a matrix with 4 rows and 1 column
y = np.array([[0, 0, 1, 1]]).T

print("output dataset:")
print(y)

# seed random numbers to make calculation deterministic
np.random.seed(1)

# randomly initialize weights with mean 0 (unsure of what this calc is)
# we have only 2 layers (input and output) so we only need one weight matrix
# to connect them (its dim is (3, 1) as we have 3 inputs and 1 output).
syn0 = 2*np.random.random((3, 1)) - 1

for i in range(10000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)

    # update weights
    syn0 += np.dot(l0.T, l1_delta)

print("Output After Training:")
print(l1)
