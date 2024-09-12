import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(1, 10, 1000)

# f1 = x*(1+((7-x**5)/x**2)**3)
# plt.plot(x, f1)
# plt.xlabel('x')
# plt.ylabel('f1')
# plt.show()

# f2 = x-(((x**5)-7)/x**2)
# plt.plot(x, f2)
# plt.xlabel('x')
# plt.ylabel('f2')
# plt.show()

# f3 = x-(((x**5)-7)/5*x**4)
# plt.plot(x, f3)
# plt.xlabel('x')
# plt.ylabel('f3')
# plt.show()
    
f4 = x-(((x**5)-7)/12)
plt.plot(x, f4)
plt.xlabel('x')
plt.ylabel('f4')
plt.show()

