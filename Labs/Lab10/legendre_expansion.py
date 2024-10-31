import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math
from scipy.integrate import quad

def driver():

#  function you want to approximate
    f = lambda x: 1/(1 + x**2)
    #f = lambda x: math.exp(x)

# Interval of interest    
    a = -1
    b = 1
# weight function    
    w = lambda x: 1.
    #Chebychev weight function:
    w_t = lambda x: 1/math.sqrt(1 - x**2)

# order of approximation
    n = 2

#  Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a,b,N+1)
    pval = np.zeros(N+1)
    pval_t = np.zeros(N+1)

    for kk in range(N+1):
      pval[kk] = eval_legendre_expansion(f,a,b,w,n,xeval[kk])
      pval_t[kk] = eval_chebychev_expansion(f,a,b,w_t,n,xeval[kk])
      
    ''' create vector with exact values'''
    fex = np.zeros(N+1)
    for kk in range(N+1):
        fex[kk] = f(xeval[kk])

    err_sum = 0
    err_sum_t = 0
    for i in range(N+1):
        err_sum = err_sum + abs(pval[i]-fex[i])
        err_sum_t = err_sum_t + abs(pval_t[i]-fex[i])
    print("Total error, Legendre:", err_sum)
    print("Total error, Chebychev:", err_sum_t)
        
    plt.figure()    
    plt.plot(xeval,fex,'ro-', label= 'f(x)')
    plt.plot(xeval,pval,'bs--',label= 'Legendre Expansion') 
    plt.plot(xeval,pval_t,'gs--',label= 'Chebychev Expansion') 
    plt.legend()
    plt.show()    
    
    err = abs(pval-fex)
    err_t = abs(pval_t-fex)
    plt.semilogy(xeval,err,'ro--',label='error, Legendre')
    plt.semilogy(xeval,err_t,'bo--',label='error, Chebychev')
    plt.legend()
    plt.show()
      
    

def eval_legendre_expansion(f,a,b,w,n,x): 

#   This subroutine evaluates the Legendre expansion

#  Evaluate all the Legendre polynomials at x that are needed
# by calling your code from prelab 
  p = eval_legendre(n,x)
  # initialize the sum to 0 
  pval = 0.0    
  for j in range(0,n+1):
      # make a function handle for evaluating phi_j(x)
      phi_j = lambda x: eval_legendre(n,x)[j]
      # make a function handle for evaluating phi_j^2(x)*w(x)
      phi_j_sq = lambda x: (eval_legendre(n,x)[j]**2)*w(x)
      # use the quad function from scipy to evaluate normalizations
      norm_fac,err = quad(phi_j_sq,a,b)
      # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
      func_j = lambda x: eval_legendre(n,x)[j]*f(x)*w(x)
      # use the quad function from scipy to evaluate coeffs
      aj,err = quad(func_j,a,b)
      #test line (divine num and denom)
      aj = aj/norm_fac
      # accumulate into pval
      pval = pval+aj*p[j] 
       
  return pval


def eval_legendre(n,x):
   p = np.zeros(n+1)
   p[0] = 1
   p[1] = x
   for i in range(1,n):
      p[i+1] = (1/i) * (((2*i + 1) * x * p[i]) - (i * p[i - 1]))
   
   return p

   
def eval_chebychev_expansion(f,a,b,w,n,x): 

#   This subroutine evaluates the Chebychev expansion

#  Evaluate all the Chebychev polynomials at x that are needed
# by calling your code from prelab 
  p = eval_chebychev(n,x)
  # initialize the sum to 0 
  pval = 0.0    
  for j in range(0,n+1):
      # make a function handle for evaluating phi_j(x)
      phi_j = lambda x: eval_chebychev(n,x)[j]
      # make a function handle for evaluating phi_j^2(x)*w(x)
      phi_j_sq = lambda x: (eval_chebychev(n,x)[j]**2)*w(x)
      # use the quad function from scipy to evaluate normalizations
      norm_fac,err = quad(phi_j_sq,a,b)
      # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
      func_j = lambda x: eval_chebychev(n,x)[j]*f(x)*w(x)
      # use the quad function from scipy to evaluate coeffs
      aj,err = quad(func_j,a,b)
      #test line (divine num and denom)
      aj = aj/norm_fac
      # accumulate into pval
      pval = pval+aj*p[j] 
       
  return pval


def eval_chebychev(n,x):
   p = np.zeros(n+1)
   p[0] = 1
   p[1] = x
   for i in range(1,n):
      p[i+1] = 2*x*p[i] - p[i-1]
   
   return p
    
if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()               
