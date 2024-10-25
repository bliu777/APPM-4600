import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm


def driver():
    
    f = lambda x: 1./(1. + x**2)
    a = -5
    b = 5
    
    #N=5
    ''' number of intervals'''
    Nint5 = 4
    Ntchint5 = Nint5 + 1
    xint5 = np.linspace(a,b,Nint5+1)
    xtchint5 = np.zeros(Ntchint5)
    for i in range(Ntchint5):
       xtchint5[i] = 5 * (np.cos(((2.*i + 1.) * np.pi)/(2. * Ntchint5)))
    yint5 = f(xint5)
    ytchint5 = f(xtchint5)

    print("xint5:", xint5)
    print("xtchint5:", xtchint5)
    print("yint5:", yint5)
    print("ytchint5:", ytchint5)

    #N=10
    ''' number of intervals'''
    Nint10 = 9
    Ntchint10 = Nint10 + 1
    xint10 = np.linspace(a,b,Nint10+1)
    xtchint10 = np.zeros(Ntchint10)
    for i in range(Ntchint10):
       xtchint10[i] = 5 * (np.cos(((2.*i + 1.) * np.pi)/(2. * Ntchint10)))
    yint10 = f(xint10)
    ytchint10 = f(xtchint10)

    #N=15
    ''' number of intervals'''
    Nint15 = 14
    Ntchint15 = Nint15 + 1
    xint15 = np.linspace(a,b,Nint15+1)
    xtchint15 = np.zeros(Ntchint15)
    for i in range(Ntchint15):
       xtchint15[i] = 5 * (np.cos(((2.*i + 1.) * np.pi)/(2. * Ntchint15)))
    yint15 = f(xint15)
    ytchint15 = f(xtchint15)

    #N=20
    ''' number of intervals'''
    Nint20 = 19
    Ntchint20 = Nint20 + 1
    xint20 = np.linspace(a,b,Nint20+1)
    xtchint20 = np.zeros(Ntchint20)
    for i in range(Ntchint20):
       xtchint20[i] = 5 * (np.cos(((2.*i + 1.) * np.pi)/(2. * Ntchint20)))
    yint20 = f(xint20)
    ytchint20 = f(xtchint20)

    ''' create points you want to evaluate at'''
    Neval = 1000
    xeval =  np.linspace(xint5[0],xint5[Nint5],Neval+1)

        
    ''' evaluate f at the evaluation points'''
    fex = f(xeval)


    '''equidistant nodes'''
    
    plt.figure()    
    plt.plot(xeval,fex,'ro-',label='exact function')
    
    #N = 5
    (M5,C5,D5) = create_natural_spline(yint5,xint5,Nint5)
    
#    print('M =', M)
#    print('C =', C)
#    print('D =', D)
    
    yeval = eval_cubic_spline(xeval,Neval,xint5,Nint5,M5,C5,D5)
    
    print('yeval = ', yeval)

    nerr5 = norm(fex-yeval)
    print('nerr5 =', nerr5)

    plt.plot(xeval,yeval,'b.--',label='natural spline, N=5') 
    
    err5 = abs(yeval-fex)

    #N=10
    (M10,C10,D10) = create_natural_spline(yint10,xint10,Nint10)
    yeval = eval_cubic_spline(xeval,Neval,xint10,Nint10,M10,C10,D10)

    nerr10 = norm(fex-yeval)
    print('nerr10 =', nerr10)

    plt.plot(xeval,yeval,'g.--',label='natural spline, N=10') 
    
    err10 = abs(yeval-fex)

    #N=15
    (M15,C15,D15) = create_natural_spline(yint15,xint15,Nint15)
    yeval = eval_cubic_spline(xeval,Neval,xint15,Nint15,M15,C15,D15)

    nerr15 = norm(fex-yeval)
    print('nerr15 =', nerr15)

    plt.plot(xeval,yeval,'m.--',label='natural spline, N=15') 
    
    err15 = abs(yeval-fex)
    
    #N=20
    (M20,C20,D20) = create_natural_spline(yint20,xint20,Nint20)
    yeval = eval_cubic_spline(xeval,Neval,xint20,Nint20,M20,C20,D20)

    nerr20 = norm(fex-yeval)
    print('nerr20 =', nerr20)

    plt.plot(xeval,yeval,'y.--',label='natural spline, N=20') 
    
    err20 = abs(yeval-fex)

    
    plt.legend()
     
    plt.figure() 
    plt.semilogy(xeval,err5,'bo--',label='absolute error')
    plt.semilogy(xeval,err10,'go--',label='absolute error')
    plt.semilogy(xeval,err15,'mo--',label='absolute error')
    plt.semilogy(xeval,err20,'yo--',label='absolute error')
    plt.legend()
    plt.show()


    '''Tchebychev nodes'''
    
    plt.figure()    
    plt.plot(xeval,fex,'ro-',label='exact function')
    
    #N = 5
    (Mtch5,Ctch5,Dtch5) = create_natural_spline(ytchint5,xtchint5,Nint5)
    
#    print('M =', M)
#    print('C =', C)
#    print('D =', D)
    
    ytcheval = eval_cubic_spline(xeval,Neval,xtchint5,Nint5,Mtch5,Ctch5,Dtch5)
    
    print('ytcheval = ', ytcheval)

    ntcherr5 = norm(fex-ytcheval)
    print('ntcherr5 =', ntcherr5)

    plt.plot(xeval,ytcheval,'b.--',label='natural spline, N=5') 
    
    errtch5 = abs(ytcheval-fex)

    #N=10
    (Mtch10,Ctch10,Dtch10) = create_natural_spline(ytchint10,xtchint10,Nint10)
    ytcheval = eval_cubic_spline(xeval,Neval,xtchint10,Nint10,Mtch10,Ctch10,Dtch10)

    ntcherr10 = norm(fex-ytcheval)
    print('ntcherr10 =', ntcherr10)

    plt.plot(xeval,ytcheval,'g.--',label='natural spline, N=10') 
    
    errtch10 = abs(ytcheval-fex)

    #N=15
    (Mtch15,Ctch15,Dtch15) = create_natural_spline(ytchint15,xtchint15,Nint15)
    ytcheval = eval_cubic_spline(xeval,Neval,xtchint15,Nint15,Mtch15,Ctch15,Dtch15)

    ntcherr15 = norm(fex-ytcheval)
    print('ntcherr15 =', ntcherr15)

    plt.plot(xeval,ytcheval,'m.--',label='natural spline, N=15') 
    
    errtch15 = abs(ytcheval-fex)
    
    #N=20
    (Mtch20,Ctch20,Dtch20) = create_natural_spline(ytchint20,xtchint20,Nint20)
    ytcheval = eval_cubic_spline(xeval,Neval,xtchint20,Nint20,Mtch20,Ctch20,Dtch20)

    ntcherr20 = norm(fex-ytcheval)
    print('ntcherr20 =', ntcherr20)

    plt.plot(xeval,ytcheval,'y.--',label='natural spline, N=20') 
    
    errtch20 = abs(ytcheval-fex)

    
    plt.legend()
     
    plt.figure() 
    plt.semilogy(xeval,errtch5,'bo--',label='absolute error')
    plt.semilogy(xeval,errtch10,'go--',label='absolute error')
    plt.semilogy(xeval,errtch15,'mo--',label='absolute error')
    plt.semilogy(xeval,errtch20,'yo--',label='absolute error')
    plt.legend()
    plt.show()
    
def create_natural_spline(yint,xint,N):

#    create the right  hand side for the linear system
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1)  
    for i in range(1,N):
       hi = xint[i]-xint[i-1]
       hip = xint[i+1] - xint[i]
       b[i] = (yint[i+1]-yint[i])/hip - (yint[i]-yint[i-1])/hi
       h[i-1] = hi
       h[i] = hip

#  create matrix so you can solve for the M values
# This is made by filling one row at a time 
    A = np.zeros((N+1,N+1))
    A[0][0] = 1.0
    for j in range(1,N):
       A[j][j-1] = h[j-1]/6
       A[j][j] = (h[j]+h[j-1])/3 
       A[j][j+1] = h[j]/6
    A[N][N] = 1

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
    print("yeval =", yeval)
    return yeval 
    
    
def  eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D):
    
    yeval = np.zeros(Neval+1)
    
    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp= xint[j+1]
        print("atmp =", atmp, ", btmp =", btmp)
#   find indices of values of xeval in the interval
        ind = np.where((xeval >= atmp) & (xeval <= btmp))
        print("ind =", ind)
        xloc = xeval[ind]
        print('xloc = ', xloc)
# evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,M[j],M[j+1],C[j],D[j])
        print('yloc = ', yloc)
#   copy into yeval
        yeval[ind] = yloc

    return(yeval)
           
driver()               

