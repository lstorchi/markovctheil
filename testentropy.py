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

filename = "entropy_100.txt"
a = 0.1

if len(sys.argv) != 3:
    print "usage: ", sys.argv[0], \
            " filename avalue" 
    exit(1)
else:
    filename = sys.argv[1] 
    a = float(sys.argv[2])

fp = open(filename)

m = []
mmax = []
mmin = []
cmpval = []

idx = 0
for l in fp:
    sval = l.split(" ")
    idx += 1

    if len(sval) == 4:
      cmpval.append(1.0 - (float(sval[2]) / a**2))
      mmax.append(float(sval[1]) + a)
      mmin.append(float(sval[1]) - a)
      m.append(float(sval[1]))
    else:
      print "error at line: ", idx 

idx = 0
for idx in range(len(m)):
    print idx+1, " ", mmin[idx], " ", m[idx], " ", mmax[idx] , " ", cmpval[idx]

fp.close()
