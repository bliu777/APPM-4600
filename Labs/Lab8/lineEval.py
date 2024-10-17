#import libraries
import numpy as np

def lineEval(x_0, y_0, x_1, y_1, a):
    lineFunc = lambda x: (y_1 - y_0) * ((x - x_0) / (x_1 - x_0)) + y_0
    return lineFunc(a)