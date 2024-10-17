# import libraries
import numpy as np
    
def driver():

# test functions 
     #fixed point is p = 1.3652300134140976
     f1 = lambda x: (10/(x+4))**(0.5)

     Nmax = 100
     tol = 1e-10

# test f1 '''
     x0 = 1.0
     returndata = fixedptallappx(f1,x0,tol,Nmax)
     for i in range(len(returndata)):
         print(returndata[i])
     x = returndata[0]
     ier = returndata[1]
     count = returndata[2]
     print('the approximate fixed point for each iteration is:',x)
     print('f1(xstar):',f1(x[count]))
     print('Error message reads:',ier)



# define routines
# def fixedpt(f,x0,tol,Nmax):

#     ''' x0 = initial guess''' 
#     ''' Nmax = max number of iterations'''
#     ''' tol = stopping tolerance'''

#     count = 0
#     while (count <Nmax):
#        count = count +1
#        x1 = f(x0)
#        if (abs(x1-x0) <tol):
#           xstar = x1
#           ier = 0
#           return [xstar,ier]
#        x0 = x1

#     xstar = x1
#     ier = 1
#     return [xstar, ier]


#new subroutines
def fixedptallappx(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    x = np.zeros((Nmax + 1,1))

    count = 0
    while (count < Nmax):
       x1 = f(x0)
       x[count] = x1
       count = count + 1
       if (abs(x1-x0) <tol):
          ier = 0
          return x,ier,count
       x0 = x1

    x[count] = x1
    ier = 1
    print("The number of iterations is:", count)
    return x,ier,count


def compute_order(x, xstar):
    ''' x = array of iterate values''' 
    ''' xstar = fixed point'''

    #|x_n+1 - x*|
    diff1 = np.abs(x[1:-1]-xstar)
    #|x_n - x*|
    diff2 = np.abs(x[0:-1]-xstar)
    #linear fit of logs to compute slope(alpha) and intercept(lambda)
    fit = np.polyfit(np.log(diff2.flatten()), np.log(diff1.flatten()), 1)

    #(x, )
    _lambda = np.exp(fit[1])
    alpha = fit[0]

    return [_lambda, alpha]
    

driver()