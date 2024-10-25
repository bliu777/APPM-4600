import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def driver():


    f = lambda x: 1./(1. + x**2)

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
   
   
    ''' create equispaced interpolation nodes'''
    xint5 = np.linspace(a,b,N5+1)
    xint10 = np.linspace(a,b,N10+1)
    xint15 = np.linspace(a,b,N15+1)
    xint20 = np.linspace(a,b,N20+1)

    print("xint5", xint5)

    ''' create Tchebychev interpolation nodes'''
    xtchint5 = np.zeros(Ntch5)
    for i in range(Ntch5):
       xtchint5[i] = 5 * (np.cos(((2.*i + 1.) * np.pi)/(2. * Ntch5)))
    
    xtchint10 = np.zeros(Ntch10)
    for i in range(Ntch10):
       xtchint10[i] = 5 * (np.cos(((2.*i + 1.) * np.pi)/(2. * Ntch10)))
       
    xtchint15 = np.zeros(Ntch15)
    for i in range(Ntch15):
       xtchint15[i] = 5 * (np.cos(((2.*i + 1.) * np.pi)/(2. * Ntch15)))

    xtchint20 = np.zeros(Ntch20)
    for i in range(Ntch20):
       xtchint20[i] = 5 * (np.cos(((2.*i + 1.) * np.pi)/(2. * Ntch20)))

       
    print("xtchint5", xtchint5)
    
    ''' create interpolation data'''
    yint5 = f(xint5)
    yint10 = f(xint10)
    yint15 = f(xint15)
    yint20 = f(xint20)

    ytchint5 = f(xtchint5)
    ytchint10 = f(xtchint10)
    ytchint15 = f(xtchint15)
    ytchint20 = f(xtchint20)
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l= np.zeros(Neval+1)
  
    ''' evaluate lagrange poly '''
    """  for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xint5,yint,N) """
    


    ''' create vector with exact values'''
    fex = f(xeval)
       

    plt.figure()    
    plt.plot(xeval,fex,'ro-',label='function')

    ''' evaluate lagrange poly N = 5'''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xint5,yint5,N5)
    plt.plot(xeval,yeval_l,'bs--',label='lagrange, N=5') 
    
    err_l5 = abs(yeval_l-fex)

    ''' evaluate lagrange poly N = 10'''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xint10,yint10,N10)
    plt.plot(xeval,yeval_l,'gs--',label='lagrange, N=10') 

    err_l10 = abs(yeval_l-fex)

    ''' evaluate lagrange poly N = 15'''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xint15,yint15,N15)
    plt.plot(xeval,yeval_l,'ms--',label='lagrange, N=15') 
    
    err_l15 = abs(yeval_l-fex)

    ''' evaluate lagrange poly N = 20'''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xint20,yint20,N20)
    plt.plot(xeval,yeval_l,'ys--',label='lagrange, N=20') 
    
    err_l20 = abs(yeval_l-fex)

    plt.legend()

    '''errors'''
    plt.figure() 
    plt.semilogy(xeval,err_l5,'bo--',label='lagrange error, N=5')
    plt.semilogy(xeval,err_l10,'go--',label='lagrange error, N=10')
    plt.semilogy(xeval,err_l15,'mo--',label='lagrange error, N=15')
    plt.semilogy(xeval,err_l20,'yo--',label='lagrange error, N=20')
    plt.legend()
    plt.show()


    '''Tchebychev Appx'''
    plt.figure()    
    plt.plot(xeval,fex,'ro-',label='function')

    ''' evaluate lagrange poly N = 5'''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xtchint5,ytchint5,N5)
    plt.plot(xeval,yeval_l,'bs--',label='lagrange w/ tchebychev nodes, N=5') 
    
    errtch_l5 = abs(yeval_l-fex)

    ''' evaluate lagrange poly N = 10'''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xtchint10,ytchint10,N10)
    plt.plot(xeval,yeval_l,'gs--',label='lagrange w/ tchebychev nodes, N=10') 

    errtch_l10 = abs(yeval_l-fex)

    ''' evaluate lagrange poly N = 15'''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xtchint15,ytchint15,N15)
    plt.plot(xeval,yeval_l,'ms--',label='lagrange w/ tchebychev nodes, N=15') 
    
    errtch_l15 = abs(yeval_l-fex)

    ''' evaluate lagrange poly N = 20'''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xtchint20,ytchint20,N20)
    plt.plot(xeval,yeval_l,'ys--',label='lagrange w/ tchebychev nodes, N=20') 
    
    errtch_l20 = abs(yeval_l-fex)

    plt.legend()

    '''errors'''
    plt.figure() 
    plt.semilogy(xeval,errtch_l5,'bo--',label='lagrange error w/ tchebychev nodes, N=5')
    plt.semilogy(xeval,errtch_l10,'go--',label='lagrange error w/ tchebychev nodes, N=10')
    plt.semilogy(xeval,errtch_l15,'mo--',label='lagrange error w/ tchebychev nodes, N=15')
    plt.semilogy(xeval,errtch_l20,'yo--',label='lagrange error w/ tchebychev nodes, N=20')
    plt.legend()
    plt.show()

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)
  

driver()
    