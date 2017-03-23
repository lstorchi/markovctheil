import numpy.linalg
import numpy.random
import scipy.stats
import scipy.io
import numpy
import math
import sys
import os

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

fp = open("entropy_100.txt")

a = 0.1
#0.05 
#0.15

for l in fp:
    print 1.0 - (float(l.split(" ")[2]) / a**2)

    print float(l.split(" ")[1]) + a
    print float(l.split(" ")[1]) - a

fp.close()
