from __future__ import division
from numpy import power, arange, exp, sum
import numpy as np
from numpy.random import choice
from scipy.misc import factorial


def geomden(x,th):
    """
    P.m.f. of the geometric distribution (size biased)
    """
    return x * power(th, (x-1)) * power(1-th,2)

def geom(th, n):
    """ generate size biased geometric offspring for th and samp size n"""

    ind = arange(1,101)
    p = [geomden(x,th) for x in ind]
    p = p / sum(p)
    
    return choice(ind, n, True, p)


def geomPoisContam(th, n, contam = .3, la = 5):
    """ generate size biased geometric offspring for th and samp size n. this is contaminated by a poisson distribution"""
    
    def foo(x,th):
        return  (1 - contam) * x * power(th, (x-1)) * (1 - th) + contam * x * power(la, x-1, dtype = np.float64) / factorial(x-1) * exp(-la)
    
    ind = arange(1,101)
    p = [foo(x,th) for x in ind]
    p = p / sum(p)
    
    return choice(ind, n, True, p)
#if __name__ == "__main__":
#    import numpy as np

def pois(th, n):
    """ generate size biased geometric offspring for th and samp size n"""

    def foo(x,th):
        return x * power(th, x-1, dtype = np.float64) / factorial(x-1) * exp(-th) / (th + 1)
    
    ind = arange(1,101)
    p = [foo(x,th) for x in ind]
    p = p / sum(p)
    
    return choice(ind, n, True, p)

def poisPoisContam(th, n, contam = .3, la = 5):
    """ generate size biased geometric offspring for th and samp size n. this is contaminated by a poisson distribution"""
    
    def foo(x,th):
        return  (1 - contam) * x * power(th, x-1, dtype = np.float64) / factorial(x-1) * exp(-th) / (th + 1) + contam * x * power(la, x-1, dtype = np.float64) / factorial(x-1) * exp(-la)
    
    ind = arange(1,101)
    p = [foo(x,th) for x in ind]
    p = p / sum(p)
    
    return choice(ind, n, True, p)
