import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def driver():


    f = lambda x: 1/(1 + (10 * x) ** 2)

    N = 18
    ''' interval'''
    a = -1
    b = 1
   
   
    ''' create equispaced interpolation nodes'''
    xint = np.linspace(a,b,N+1)
    
    ''' create interpolation data'''
    yint = f(xint)
    
    ''' create points for evaluating the Barycentric interpolating polynomial'''
    Neval = 1001
    xeval = np.linspace(a,b,Neval+1)
    yeval= np.zeros(Neval+1)

    ''' evaluate barycentric poly '''
    for kk in range(Neval+1):
       yeval[kk] = eval_barycentric(xeval[kk],xint,yint,N)


    ''' create vector with exact values'''
    fex = f(xeval)
       

    plt.figure()    
    plt.plot(xeval,fex,'ro-')
    plt.plot(xeval,yeval,'bs--') 
    plt.legend()

    plt.figure() 
    err = abs(yeval-fex)
    plt.semilogy(xeval,err,'ro--',label='barycentric')
    plt.legend()
    plt.show()

def eval_barycentric(xeval,xint,yint,N):

    l = 1
    wj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
            if (jj != count):
                wj[jj] = wj[jj] / (xint[count]-xint[jj])

    for i in range(N+1):
        l = l*(xeval - xint[i])

    yeval = 0.
        
    for j in range(N+1):
        yeval = yeval + ((wj[j] * yint[j]) / (xeval - xint[j]))

    yeval = yeval * l

    """ lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])
           else:
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    
    for jj in range(N+1):
       yeval = yeval + (yint[jj]*lj[jj]) / (xeval - xint[jj]) """
  
    return(yeval)
       

driver()        
