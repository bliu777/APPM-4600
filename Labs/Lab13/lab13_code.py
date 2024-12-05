import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
import time

def driver():
    ''' create matrix for testing different ways of solving a square
    linear system'''
    '''' N = size of system'''
    N = 8000
    ''' Right hand side'''
    b = np.random.rand(N,1)
    A = np.random.rand(N,N)

    startTime = time.time()
    x = scila.solve(A,b)
    endTime = time.time()
    totalTime = endTime - startTime

    test = np.matmul(A,x)
    r = la.norm(test-b)
    print(r, 'time:', totalTime)


    #startTime = time.time()
    x, luTime, solveTime = lu_factor(A,b)
    #endTime = time.time()
    #totalTime = endTime - startTime

    test = np.matmul(A,x)
    r = la.norm(test-b)
    print(r, 'lu compute time:', luTime, 'solve time:', solveTime, 'total time:', luTime + solveTime)

    ''' Create an ill-conditioned rectangular matrix '''
    N = 10
    M = 5
    A = create_rect(N,M)
    b = np.random.rand(N,1)




def create_rect(N,M):
    ''' this subroutine creates an ill-conditioned rectangular matrix'''
    a = np.linspace(1,10,M)
    d = 10**(-a)
    D2 = np.zeros((N,M))
    for j in range(0,M):
        D2[j,j] = d[j]

    '''' create matrices needed to manufacture the low rank matrix'''
    A = np.random.rand(N,N)
    Q1, R = la.qr(A)
    test = np.matmul(Q1,R)
    A = np.random.rand(M,M)
    Q2,R = la.qr(A)
    test = np.matmul(Q2,R)
    B = np.matmul(Q1,D2)
    B = np.matmul(B,Q2)
    return B

def lu_factor(A,b):
    startTime = time.time()
    lu,p = scila.lu_factor(A)
    endTime = time.time()
    luTime = endTime - startTime

    startTime = time.time()
    x = scila.lu_solve((lu,p), b)
    endTime = time.time()
    solveTime = endTime - startTime

    return x, luTime, solveTime

if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()
