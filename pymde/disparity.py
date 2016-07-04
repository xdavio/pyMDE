#import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from est import kernden
from est import modellikelihood

from estPois import modellikelihood as poismodel

from utils import unbound_access

import autograd.numpy as np
from autograd import grad


class RAF():
    def __init__(self, cfun):
        if isinstance(cfun, Cfun):
            self.cfun = cfun._cfun
        else:
            self.cfun = cfun
        self.Cprime = grad(self.cfun)
        
        
    def eval(self, x):
        return self.Cprime(x) * (1 + x) - self.cfun(x)

class RAF_C(RAF):
    def __init__(self, Cfun_inst):
        """
        this RAF class takes an instance of a CFUN instead of the class
        """
    

class Cfun():
    """ This is the object which is minimized according to standard disparity theory. """
    def __init__(self, d, th = .3, likelihood = None):
        self.k = 100 #support
        self.th = th #initial guess of theta
        self.d = d #data
        self.overflow = .001
        if likelihood is None:
            self.modellikelihood = modellikelihood
        else:
            self.modellikelihood = likelihood
        
    def resid(self, th, overflowprotect = True):
        """
        Pearson residual
        """
        #d is the data
        #k is the max value of the support
        out = kernden(self.d, self.k) / self.modellikelihood(self.k, th) - 1
        delta = np.zeros(self.k)
        if overflowprotect:
            for i in np.arange(self.k):
                if out[i] == -1:
                    delta[i] = self.overflow
        return(out + delta)
                    

    def _cfun(self, d):
        """
        This function returns the value of C.
        """
        pass
    
    def cfun(self, th):
        d = self.resid(th) #delta
        distance = self._cfun(d)
        return(distance)

    def rho(self, th):
        return(np.array(self.cfun(th)) * self.modellikelihood(self.k, th))

    def obj(self, th):
        return(np.sum(self.rho(th)))

    def mde(self, lower = 0, upper = 1):
        out = minimize_scalar(fun = self.obj, bounds = (lower,upper), method = "Bounded")
        return(out.x)


class LD(Cfun):
    @unbound_access
    def _cfun(self, d):
        return (d + 1) * np.log(d + 1) - d

class HD(Cfun):
    @unbound_access
    def _cfun(self, d):
        return 2 * np.power( np.power( d + 1 , .5) - 1, 2)

class PCS(Cfun):
    @unbound_access
    def _cfun(self, d):
        return np.power(d,2) / 2

class NCS(Cfun):
    @unbound_access
    def _cfun(self, d):
        return np.power(d,2) / (2 * (d + 1))

class KLD(Cfun):
    @unbound_access
    def _cfun(self, d):
        return d - np.log(d + 1)

class SCS(Cfun):
    @unbound_access
    def _cfun(self, d):
        return np.power(d,2) / (d + 2)

class NED(Cfun):
    @unbound_access
    def _cfun(self, d):
        return np.exp(-d) - 1 + d

class PD(Cfun):
    @unbound_access
    def _cfun(self, d, pd_la = 1):
        return (np.power(d + 1, pd_la + 1) - (d +1))/(pd_la * (pd_la + 1)) - d / (pd_la + 1)

def mde(d, clss, likelihood = None, bounds = (0,1)):
    """
    Takes class pointer clss and dataset d and returns the MDE for that class
    """
    a = clss(d, .3, likelihood)
    return(a.mde(*bounds))


if __name__ == "main":
    b = LD(a)
    c = HD(a)
    d = PCS(a)
    e = NCS(a)
    f = KLD(a)
    b.rho()
    c.rho()
    d.rho()
    e.rho()
    f.rho()
