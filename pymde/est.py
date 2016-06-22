import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

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
    return foo / sum(foo)


def modellikelihood(k,theta):
    #hardcoded geometric dist shifted by +1
    #def foo(x,theta):
    #    return np.power(theta,x) * (1-theta) / theta
    #return np.array([foo(y,theta) for y in range(1,k+1)])
    return theta ** np.arange(1,k+1) * (1 - theta) / theta

    
def objective(theta,d,k):
    #theta - parameter of interest
    #x - the observed brood sizes
    #k - the indices of summation in the objective function
    return -np.sum(
        np.power(
            modellikelihood(k,theta)*kernden(d,k),
            .5
            )
        )


def penalized_hell_objective(theta,d,k,h):
    #find those points d with 0
    notd = [x for x in np.arange(1,kmax + 1) if x not in np.unique(d)]
    foo = modellikelihood(k,theta) #likelihood cache

    return objective(theta,d,k) + (h - 2) * np.sum( [foo[x-1] for x in notd] ) / 4


def mhde(d):
    out = minimize_scalar(fun = objective, args = (d, kmax), bounds = (0,1), method = "Bounded")
    return out.x

def pmhde(d, h = 1):
    out = minimize_scalar(fun = penalized_hell_objective, args = (d, kmax, h), bounds = (0,1), method = "Bounded")
    return out.x


def aml(d):
    foo = np.mean(d)
    return ( foo - 1 ) / ( foo + 1 )


def amm(d):
    foo = 1./(np.mean(np.power(d,-1.))) #this is for mu of offspring
    return (foo - 1.) / foo


#asymptotic standard deviations
def ammsd(th):
    return np.power( - np.power(1 - th, 2) * ( np.log(1 - th) / th  + 1) , .5)


def amlsd(th):
    return np.power((th * np.power( 1 - th, 2) / 2), .5)


## def mhdesd(th):    
##     def mhdeasvar(th):
##         def mhdeasvarnum(th):
##             def den(x):
##                 return np.power(th,(x-1)) * (1-th)            
##             X = np.arange(1,mhdesupport + 1)
##             mu = 1. / (1 - th)
##             return mu * np.sum(  np.power( (X-1)/th - 1./(1-th), 2) / X * den(X) )        
##         f = mhdeasvarnum(th)
##         information = 1./(th*np.power(1-th,2)) #information of fth not fth,app
##         return f/np.power(information, 2)    
##     return np.power(mhdeasvar(th), .5)

def mhdesd(th):
    #for geometri, mhdesd is the same as ammsd
    return ammsd(th)

def pmhdesd(th):
    #for geometric, mhdesd is the same as ammsd
    return 0




