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

if len(sys.argv) == 3:
    filename = sys.argv[1]
    rownum = int(sys.argv[2])
else:
    print "usage: ", sys.argv[0], " ratingmtx row_number"
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


fp = open("change.txt", "w")

maxval = -1.0 * float("inf")
cp = 0
for c_p in range(1, time):
    L, L1, L2 = changemod.compute_ls(rm, c_p, rownum, errmsg)
    
    if (L == None):
        print errmsg
        exit(1)

    if (maxval < L1+L2):
        maxval = L1 + L2
        cp = c_p
    
    fp.write(str(c_p) + " " + str(L1+L2) + "\n")

fp.close()
print cp, maxval
