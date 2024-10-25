import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math

def driver():


    f = lambda x: 1./(1.+x**2)
    fp = lambda x: -2*x/(1.+x**2)**2

    N5 = 4
    N10 = 9
    N15 = 14
    N20 = 19

    Ntch5 = N5+1
    Ntch10 = N10+1
    Ntch15 = N15+1
    Ntch20 = N20+1

    ''' interval'''
    a = -5
    b = 5
   
    '''equispaced'''
#   N=5
    ''' create equispaced interpolation nodes'''
    xint5 = np.linspace(a,b,N5+1)
    
    ''' create interpolation data'''
    yint5 = np.zeros(N5+1)
    ypint5 = np.zeros(N5+1)
    for jj in range(N5+1):
        yint5[jj] = f(xint5[jj])
        ypint5[jj] = fp(xint5[jj])

#   N=10
    ''' create equispaced interpolation nodes'''
    xint10 = np.linspace(a,b,N10+1)
    
    ''' create interpolation data'''
    yint10 = np.zeros(N10+1)
    ypint10 = np.zeros(N10+1)
    for jj in range(N10+1):
        yint10[jj] = f(xint10[jj])
        ypint10[jj] = fp(xint10[jj])
        
#   N=15
    ''' create equispaced interpolation nodes'''
    xint15 = np.linspace(a,b,N15+1)
    
    ''' create interpolation data'''
    yint15 = np.zeros(N15+1)
    ypint15 = np.zeros(N15+1)
    for jj in range(N15+1):
        yint15[jj] = f(xint15[jj])
        ypint15[jj] = fp(xint15[jj])

#   N=20
    ''' create equispaced interpolation nodes'''
    xint20 = np.linspace(a,b,N20+1)
    
    ''' create interpolation data'''
    yint20 = np.zeros(N20+1)
    ypint20 = np.zeros(N20+1)
    for jj in range(N20+1):
        yint20[jj] = f(xint20[jj])
        ypint20[jj] = fp(xint20[jj])

    '''Tchebychev'''
    #   N=5
    ''' create Tchebychev interpolation nodes'''
    xtchint5 = np.zeros(Ntch5)
    for i in range(Ntch5):
       xtchint5[i] = 5 * (np.cos(((2.*i + 1.) * np.pi)/(2. * Ntch5)))
    
    ''' create interpolation data'''
    ytchint5 = np.zeros(Ntch5)
    yptchint5 = np.zeros(Ntch5)
    for jj in range(Ntch5):
        ytchint5[jj] = f(xtchint5[jj])
        yptchint5[jj] = fp(xtchint5[jj])

#   N=10
    ''' create Tchebychev interpolation nodes'''
    xtchint10 = np.zeros(Ntch10)
    for i in range(Ntch10):
       xtchint10[i] = 5 * (np.cos(((2.*i + 1.) * np.pi)/(2. * Ntch10)))
    
    ''' create interpolation data'''
    ytchint10 = np.zeros(Ntch10)
    yptchint10 = np.zeros(Ntch10)
    for jj in range(Ntch10):
        ytchint10[jj] = f(xtchint10[jj])
        yptchint10[jj] = fp(xtchint10[jj])
        
#   N=15
    ''' create Tchebychev interpolation nodes'''
    xtchint15 = np.zeros(Ntch15)
    for i in range(Ntch15):
       xtchint15[i] = 5 * (np.cos(((2.*i + 1.) * np.pi)/(2. * Ntch15)))
    
    ''' create interpolation data'''
    ytchint15 = np.zeros(Ntch15)
    yptchint15 = np.zeros(Ntch15)
    for jj in range(Ntch15):
        ytchint15[jj] = f(xtchint15[jj])
        yptchint15[jj] = fp(xtchint15[jj])

#   N=20
    ''' create Tchebychev interpolation nodes'''
    xtchint20 = np.zeros(Ntch20)
    for i in range(Ntch20):
       xtchint20[i] = 5 * (np.cos(((2.*i + 1.) * np.pi)/(2. * Ntch20)))
    
    ''' create interpolation data'''
    ytchint20 = np.zeros(Ntch20)
    yptchint20 = np.zeros(Ntch20)
    for jj in range(Ntch20):
        ytchint20[jj] = f(xtchint20[jj])
        yptchint20[jj] = fp(xtchint20[jj])
    
    ''' create points for evaluating the Hermite interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yevalH = np.zeros(Neval+1)

    ''' create vector with exact values'''
    fex = np.zeros(Neval+1)
    for kk in range(Neval+1):
        fex[kk] = f(xeval[kk])
    
    
    plt.figure()
    plt.plot(xeval,fex,'ro-')

    '''equispaced'''
    #N=5
    for kk in range(Neval+1):
      yevalH[kk] = eval_hermite(xeval[kk],xint5,yint5,ypint5,N5)
    plt.plot(xeval,yevalH,'b.--',label='Hermite, N=5')
    
    errH5 = abs(yevalH-fex)

    #N=10
    for kk in range(Neval+1):
      yevalH[kk] = eval_hermite(xeval[kk],xint10,yint10,ypint10,N10)
    plt.plot(xeval,yevalH,'g.--',label='Hermite, N=10')
    
    errH10 = abs(yevalH-fex)

    #N=15
    for kk in range(Neval+1):
      yevalH[kk] = eval_hermite(xeval[kk],xint15,yint15,ypint15,N15)
    plt.plot(xeval,yevalH,'m.--',label='Hermite, N=15')
    
    errH15 = abs(yevalH-fex)

    #N=20
    for kk in range(Neval+1):
      yevalH[kk] = eval_hermite(xeval[kk],xint20,yint20,ypint20,N20)
    plt.plot(xeval,yevalH,'y.--',label='Hermite, N=20')
    
    errH20 = abs(yevalH-fex)

    
    plt.semilogy()
    plt.legend()
         
    plt.figure()
    plt.semilogy(xeval,errH5,'b.--',label='Hermite error, N=5')
    plt.semilogy(xeval,errH10,'g.--',label='Hermite error, N=10')
    plt.semilogy(xeval,errH15,'m.--',label='Hermite error, N=15')
    plt.semilogy(xeval,errH20,'y.--',label='Hermite error, N=20')
    plt.legend()
    plt.show()         


    '''Tchebychev Nodes'''

    plt.figure()
    plt.plot(xeval,fex,'ro-')

    
    #N=5
    for kk in range(Neval+1):
      yevalH[kk] = eval_hermite(xeval[kk],xtchint5,ytchint5,yptchint5,N5)
    plt.plot(xeval,yevalH,'b.--',label='Hermite Tchebychev, N=5')
    
    errtchH5 = abs(yevalH-fex)

    #N=10
    for kk in range(Neval+1):
      yevalH[kk] = eval_hermite(xeval[kk],xtchint10,ytchint10,yptchint10,N10)
    plt.plot(xeval,yevalH,'g.--',label='Hermite Tchebychev, N=10')
    
    errtchH10 = abs(yevalH-fex)

    #N=15
    for kk in range(Neval+1):
      yevalH[kk] = eval_hermite(xeval[kk],xtchint15,ytchint15,yptchint15,N15)
    plt.plot(xeval,yevalH,'m.--',label='Hermite Tchebychev, N=15')
    
    errtchH15 = abs(yevalH-fex)

    #N=20
    for kk in range(Neval+1):
      yevalH[kk] = eval_hermite(xeval[kk],xtchint20,ytchint20,yptchint20,N20)
    plt.plot(xeval,yevalH,'y.--',label='Hermite Tchebychev, N=20')
    
    errtchH20 = abs(yevalH-fex)

    
    plt.semilogy()
    plt.legend()
         
    plt.figure()
    plt.semilogy(xeval,errtchH5,'b.--',label='Hermite Tchebychev error, N=5')
    plt.semilogy(xeval,errtchH10,'g.--',label='Hermite Tchebychev error, N=10')
    plt.semilogy(xeval,errtchH15,'m.--',label='Hermite Tchebychev error, N=15')
    plt.semilogy(xeval,errtchH20,'y.--',label='Hermite Tchebychev error, N=20')
    plt.legend()
    plt.show()

def eval_hermite(xeval,xint,yint,ypint,N):

    ''' Evaluate all Lagrange polynomials'''

    lj = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    ''' Construct the l_j'(x_j)'''
    lpj = np.zeros(N+1)
#    lpj2 = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
#              lpj2[count] = lpj2[count]*(xint[count] - xint[jj])
              lpj[count] = lpj[count]+ 1./(xint[count] - xint[jj])
              

    yeval = 0.
    
    for jj in range(N+1):
       Qj = (1.-2.*(xeval-xint[jj])*lpj[jj])*lj[jj]**2
       Rj = (xeval-xint[jj])*lj[jj]**2
#       if (jj == 0):
#         print(Qj)
         
#         print(Rj)
#         print(Qj)
#         print(xeval)
 #        return
       yeval = yeval + yint[jj]*Qj+ypint[jj]*Rj
       
    return(yeval)
       

    

       
if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()        
