#import relevant libraries
import numpy as np
import matplotlib.pyplot as plt

#create identical sets of integers from 1-10
X = np.linspace(1, 10, 10)
Y = np.arange(1, 10, 1)

#Print X and Y to verify they're the same
print(X)
print(Y)

#Print the first 3 values of X
print('The first 3 values of X are: ', X[0], X[1], X[2])


w = 10**(-np.linspace(1,10,10))

print(w)
