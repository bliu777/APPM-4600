import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm

def driver():
    f = lambda x: np.sin(x)
    p33 = lambda x: (x - ((7 * x**3) / 60)) / (1. + ((x**2) / 20))
    p24 = lambda x: x / (1. + ((x**2) / 6) + ((7 * x**4) / 360))
    p42 = lambda x: (x - ((7 * x**3) / 60)) / (1. + ((x**2) / 20))
    m6 = lambda x: x - (x**3 / 6) + (x**5 / 120)

    a = 0
    b = 5

    ''' create points you want to evaluate at'''
    Neval = 1000
    xeval =  np.linspace(a,b,Neval+1)

    ''' evaluate f at the evaluation points'''
    fex = f(xeval)

    plt.figure()
    #p33
    yeval = p33(xeval)
    err = abs(fex-yeval)
    plt.semilogy(xeval,err,'b.--',label='P^3_3 error')
    #p24
    yeval = p24(xeval)
    err = abs(fex-yeval)
    plt.semilogy(xeval,err,'g.--',label='P^2_4 error')
    #p42
    yeval = p42(xeval)
    err = abs(fex-yeval)
    plt.semilogy(xeval,err,'m.--',label='P^4_2 error')
    #m6
    yeval = m6(xeval)
    err = abs(fex-yeval)
    plt.semilogy(xeval,err,'y.--',label='M_6 error')

    plt.legend()
    plt.show()  
    





driver()