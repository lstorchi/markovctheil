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
c_p = 0 

if len(sys.argv) == 3:
    filename = sys.argv[1]
    c_p = int(sys.argv[2])
else:
    print "usage: ", sys.argv[0], " ratingmtx changepoint"
    exit()
 
msd = scipy.io.loadmat(filename)

namems = "ratings"

if not(namems in msd.keys()):
    print ("Cannot find " + namems + " in " + filename)
    print (msd.keys())
    exit(1)

rm = msd[namems]

countries=rm.shape[0]
rating=numpy.max(rm)
time=rm.shape[1]

errmsg = ""
L, L1, L2, pr1 = changemod.compute_ls(rm, c_p, errmsg)

if (L == None):
    print errmsg
    exit(1)
summa = L1 + L2

print L, L1, L2, summa 
lamda = 2.0*((L1+L2)-L)

maxrat = -1.0 * float("inf")
minrat = float("inf")
for c in range(countries):
    for t in range(time):
        if (maxrat < rm[c, t]):
            maxrat = rm[c, t]
        if (minrat > rm[c, t]):
            minrat = rm[c, t]

ndof = (maxrat - minrat + 1) * (maxrat - minrat)

chi2 = scipy.stats.chi2.isf(0.05, ndof)
pvalue = 1.0 - scipy.stats.chi2.cdf(lamda, ndof)

print "Lamda   : ", lamda
print "Chi2    : ", chi2
print "P-Value : ", pvalue
plt.plot(rm[6,:])
plt.show()
