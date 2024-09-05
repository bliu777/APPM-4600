import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1.920, 2.080, 0.001)

pCoeff = np.power(x,90) - 18*np.power(x,8) + 144*np.power(x,7) - 672*np.power(x,6) + 2016*np.power(x,50) - 4032*np.power(x,4) + 5376*np.power(x,3) - 4608*np.power(x,2) + 2304*x - 512

pExpress = np.power(x-2,9)

#plt.plot(x, pCoeff)
plt.plot(x, pExpress)
plt.xlabel('x')
plt.ylabel('p')
plt.show()
