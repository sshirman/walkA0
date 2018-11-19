import sys
import numpy as np
import scipy as sp
import sympy as sym
import matplotlib.pyplot as plt
import os

def ParDeriv(x,p):
    d1 = (x[1] - x[9])*x[10] - x[0] + p
    d2 = (x[2] - x[10])*x[0] - x[1] + p
    d3 = (x[3] - x[0])*x[1] - x[2] + p
    d4 = (x[4] - x[1])*x[2] - x[3] + p
    d5 = (x[5] - x[2])*x[3] - x[4] + p
    d6 = (x[6] - x[3])*x[4] - x[5] + p
    d7 = (x[7] - x[4])*x[5] - x[6] + p
    d8 = (x[8] - x[5])*x[6] - x[7] + p
    d9 = (x[9] - x[6])*x[7] - x[8] + p
    d10 = (x[10] - x[7])*x[8] - x[9] + p
    d11 = (x[0] - x[8])*x[9] - x[10] + p
    return np.array([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11])
