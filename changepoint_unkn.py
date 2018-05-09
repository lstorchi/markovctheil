import scipy
import numpy
import math
import sys

import scipy.stats
import scipy.io

import matplotlib.pyplot as plt

sys.path.append("./module")
import basicutils
import changemod

filename = ""

if len(sys.argv) == 2:
    filename = sys.argv[1]
else:
    print "usage: ", sys.argv[0], " ratingmtx"
    exit()
 
msd = scipy.io.loadmat(filename)

namems = "ratings"

if not(namems in msd.keys()):
    print ("Cannot find " + namems + " in " + filename)
    print (msd.keys())
    exit(1)

rm = msd[namems]
countries = rm.shape[0]

"""
rmi = msd[namems]
countries = rmi.shape[0]
countries = countries - 1

rm = numpy.zeros((countries, rmi.shape[1]), dtype='int64')

idx = 0
for i in range(countries + 1):
    if i != 6:
        for j in range(rmi.shape[1]):
           rm[idx, j] = rmi[i, j]

        idx = idx + 1
"""

rating=numpy.max(rm)
time=rm.shape[1]

errmsg = ""

fp = open("change.txt", "w")

maxval = -1.0 * float("inf")
cp = 0
for c_p in range(1, time):
    print c_p , " of ", time-1
    L, L1, L2, pr1 = changemod.compute_ls(rm, c_p, errmsg)
    
    if (L == None):
        print errmsg
        exit(1)

    if (maxval < L1+L2):
        maxval = L1 + L2
        cp = c_p
    
    fp.write(str(c_p) + " " + str(L1+L2) + "\n")

fp.close()
print cp, maxval
