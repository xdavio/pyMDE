import numpy as np

#decorator which rounds to 2 decimal places
def pretty(est_fun):
    def pretty_out(*args):
        return round(est_fun(*args), 2)
        #return est_fun(*args)
    return pretty_out

#descriptor for handling unbound access to method _cfun in class Cfun
class unbound_access(object):
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls=None):
        if instance is None:
            return classmethod(self.func).__get__(None, cls)
        return self.func.__get__(instance, cls)
