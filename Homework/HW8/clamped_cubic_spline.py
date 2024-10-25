import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm


def driver():
    
    f = lambda x: 1./(1. + x**2)
    fp = lambda x: -2*x/(1.+x**2)**2
    a = -5
    b = 5
    
    
    #N=5
    ''' number of intervals'''
    Nint5 = 4
    xint5 = np.linspace(a,b,Nint5+1)
    yint5 = f(xint5)
    ypint5 = fp(xint5)

    #N=10
    ''' number of intervals'''
    Nint10 = 9
    xint10 = np.linspace(a,b,Nint10+1)
    yint10 = f(xint10)
    ypint10 = fp(xint10)

    #N=15
    ''' number of intervals'''
    Nint15 = 14
    xint15 = np.linspace(a,b,Nint15+1)
    yint15 = f(xint15)
    ypint15 = fp(xint15)

    #N=20
    ''' number of intervals'''
    Nint20 = 19
    xint20 = np.linspace(a,b,Nint20+1)
    yint20 = f(xint20)
    ypint20 = fp(xint20)


    ''' create points you want to evaluate at'''
    Neval = 1000
    xeval =  np.linspace(xint5[0],xint5[Nint5],Neval+1)

    ''' evaluate f at the evaluation points'''
    fex = f(xeval)

    
    plt.figure()    
    plt.plot(xeval,fex,'ro-',label='exact function')
    
    #N = 5
    (M,C,D) = create_clamped_spline(yint5,ypint5,xint5,Nint5)
    
#    print('M =', M)
#    print('C =', C)
#    print('D=', D)
    
    yeval = eval_cubic_spline(xeval,Neval,xint5,Nint5,M,C,D)
    
#    print('yeval = ', yeval)

    nerr5 = norm(fex-yeval)
    print('nerr5 = ', nerr5)

    plt.plot(xeval,yeval,'b.--',label='natural spline, N=5') 

    #N=10
    (M,C,D) = create_clamped_spline(yint10,ypint10,xint10,Nint10)
    yeval = eval_cubic_spline(xeval,Neval,xint10,Nint10,M,C,D)

    nerr10 = norm(fex-yeval)
    print('nerr10 = ', nerr10)

    plt.plot(xeval,yeval,'g.--',label='natural spline, N=10') 

    #N=15
    (M,C,D) = create_clamped_spline(yint15,ypint15,xint15,Nint15)
    yeval = eval_cubic_spline(xeval,Neval,xint15,Nint15,M,C,D)

    nerr15 = norm(fex-yeval)
    print('nerr15 = ', nerr15)

    plt.plot(xeval,yeval,'m.--',label='natural spline, N=15') 
    
    #N=20
    (M,C,D) = create_clamped_spline(yint20,ypint20,xint20,Nint20)
    yeval = eval_cubic_spline(xeval,Neval,xint20,Nint20,M,C,D)

    nerr20 = norm(fex-yeval)
    print('nerr20 = ', nerr20)

    plt.plot(xeval,yeval,'y.--',label='natural spline, N=20') 

    
    plt.legend()
    plt.show()
    
def create_clamped_spline(yint,ypint,xint,N):

#    create the right  hand side for the linear system
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1)

    h0 = xint[1] - xint[0]
    b[0] = -ypint[0] + (yint[1] - yint[0])/h0
    for i in range(1,N):
       hi = xint[i]-xint[i-1]
       hip = xint[i+1] - xint[i]
       b[i] = (yint[i+1]-yint[i])/hip - (yint[i]-yint[i-1])/hi
       h[i-1] = hi
       h[i] = hip
    b[N] = -ypint[N] + (yint[N] - yint[N-1])/h[N-1]

#  create matrix so you can solve for the M values
# This is made by filling one row at a time 
    A = np.zeros((N+1,N+1))
    A[0][0] = h[0]/3
    A[0][1] = h[0]/6
    for j in range(1,N):
       A[j][j-1] = h[j-1]/6
       A[j][j] = (h[j]+h[j-1])/3 
       A[j][j+1] = h[j]/6
    A[N][N-1] = h[N-1]/6
    A[N][N] = h[N-1]/3

    Ainv = inv(A)
    
    M  = Ainv.dot(b)

#  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
       C[j] = yint[j]/h[j]-h[j]*M[j]/6
       D[j] = yint[j+1]/h[j]-h[j]*M[j+1]/6
    return(M,C,D)
       
def eval_local_spline(xeval,xi,xip,Mi,Mip,C,D):
# Evaluates the local spline as defined in class
# xip = x_{i+1}; xi = x_i
# Mip = M_{i+1}; Mi = M_i

    hi = xip-xi
    yeval = (Mi*(xip-xeval)**3 +(xeval-xi)**3*Mip)/(6*hi) \
            + C*(xip-xeval) + D*(xeval-xi)
    return yeval 
    
    
def eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D):
    
    yeval = np.zeros(Neval+1)
    
    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp= xint[j+1]
        
#   find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

# evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,M[j],M[j+1],C[j],D[j])
#        print('yloc = ', yloc)
#   copy into yeval
        yeval[ind] = yloc

    return(yeval)
           
driver()               

