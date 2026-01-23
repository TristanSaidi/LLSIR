import numpy as np

def link_id(t):
    return t

def quadratic(t):
    return t ** 2

def nonsmooth_lipschitz(t, L=1.0):
    # nonsmooth Lipschitz function with Lipschitz constant L
    if t < 0.5:
        return L * t
    else:
        return L * (1 - t) + L * 0.5
