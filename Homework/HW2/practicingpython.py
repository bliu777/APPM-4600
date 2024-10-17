import numpy as np
import matplotlib.pyplot as plt

#a)
t = np.arange(0, np.pi, np.pi/30)
y = np.cos(t)

N = t.size

S = 0
for i in range(N):
    S += t[i] * y[i]

print("the sum is:", S)


#b)
