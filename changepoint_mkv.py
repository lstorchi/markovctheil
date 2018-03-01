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
rownum = 0
c_p = 0 

if len(sys.argv) == 4:
    filename = sys.argv[1]
    rownum = int(sys.argv[2])
    c_p = int(sys.argv[3])
else:
    print "usage: ", sys.argv[0], " ratingmtx row_number changepoint"
    exit()
 
msd = scipy.io.loadmat(filename)

namems = "ratings"

if not(namems in msd.keys()):
    print ("Cannot find " + namems + " in " + filename)
    print (msd.keys())
    exit(1)

rm = msd[namems]

countries=rm.shape[0]

if (rownum >= countries):
    print "Max rownum value should be ", countries-1 
    exit(1)

rating=numpy.max(rm)
time=rm.shape[1]

errmsg = ""
L, L1, L2 = changemod.compute_ls(rm, c_p, rownum, errmsg)

if (L == None):
    print errmsg
    exit(1)

#print L, L1, L2
lamda = -2*((L1+L2)-L)
print "Lamda: ", lamda


maxrat = -1.0 * float("inf")
minrat = float("inf")
for t in range(time):
    if (maxrat < rm[rownum, t]):
        maxrat = rm[rownum, t]
    if (minrat > rm[rownum, t]):
        minrat = rm[rownum, t]

ndof = (maxrat - minrat + 1) * (maxrat - minrat)

chi2 = scipy.stats.chi2.isf(0.05, ndof)
print "Chi2", chi2
