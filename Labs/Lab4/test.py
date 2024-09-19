import numpy as np

def compute_order(x, xstar):

    #|x_n+1 - x*|
    diff1 = np.abs(x[1::]-xstar)
    #|x_n - x*|
    diff2 = np.abs(x[0:-1]-xstar)
    #linear fit of logs to compute slope(alpha) and intercept(lambda)
    fit = np.polyfit(np.log(diff2.flatten()), np.log(diff1.flatten()), 1)

    #(x, )
    _lambda = np.exp(fit[1])
    alpha = fit[0]

    return [_lambda, alpha]