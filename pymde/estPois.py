import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from scipy.misc import factorial

#globals
mhdesupport = 100 #asymp var of mhde support max
kmax = 100 #objective fun suppport max

#estimates
def kernden(d,k):
    """kernel density. k is max support value and d is data. returns 100-vec of mass"""
    #foo = np.bincount(d.astype(int), weights = 1./d , minlength = k)
    foo = np.bincount(d.astype(int), minlength = k + 1)
    foo = np.delete(foo,0)
    foo = foo.astype(np.float64) / np.arange(1, k+1)
    #return foo / sum(foo)
    return foo


def modellikelihood(k,theta):
    #hardcoded geometric dist shifted by +1
    #def foo(x,theta):
    #    return np.power(theta,x) * (1-theta) / theta
    #return np.array([foo(y,theta) for y in range(1,k+1)])
    return 1. / factorial( np.arange(1,k+1) - 1 ) * theta ** ( np.arange(1,k+1) - 1) * np.exp(-theta) 


def penalized_hell_objective(theta,d,k,h):
    #find those points d with 0
    notd = [x for x in np.arange(1,kmax + 1) if x not in np.unique(d)]
    foo = modellikelihood(k,theta) #likelihood cache

    return objective(theta,d,k) + (h - 2) * np.sum( [foo[x-1] for x in notd] ) / 4


def objective(theta,d,k):
    #theta - parameter of interest
    #x - the observed brood sizes
    #k - the indices of summation in the objective function
    return -np.sum(np.power(modellikelihood(k,theta)*kernden(d,k), .5))


def mhde(d):
    out = minimize_scalar(fun = objective, args = (d, kmax), bounds = (0,100), method = "Bounded")
    return out.x

def pmhde(d, h = 1):
    out = minimize_scalar(fun = penalized_hell_objective, args = (d, kmax, h), bounds = (0,100), method = "Bounded")
    return out.x


def aml(d):
    foo = np.mean(d)
    return 1./2 * (foo - 3 + np.power(5 - 2 * foo + foo ** 2 , .5))


def amm(d):
    foo = 1./(np.mean(np.power(d,-1.))) #this is for mu of offspring
    return foo - 1


#asymptotic standard deviations
def ammsd(th):
    foo = (th + 1) ** 2 / th * ( 1 - (th + 1) * np.exp( -th ))
    return np.power(foo,.5)


def amlsd(th):
    sig = th + th/(1 + th)**2
    h = (th + 1)**2 / (th * (th + 2) + 2)
    return np.power(sig * h**2,.5)


def mhdesd(th):    
    X = np.arange(1,mhdesupport + 1)
    foo = th * (th + 1) * np.sum( ( (X-1)/th - 1 )**2 / factorial(X) * np.power(th, X,dtype = np.float64) ) * np.exp( -th )
    return np.power(foo,.5)

#ci contain

