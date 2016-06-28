import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol, lambdify

def plot_c(func, id, leftlim = -1.0, rightlim = 5.0):
    ax = plt.subplot(id)
    
    t = np.arange(leftlim, rightlim, 0.01)
    try:
        s = func(t)
    except:
        s = [func(x) for x in t]
    line, = plt.plot(t, s, lw=2)
    
    plt.show()

