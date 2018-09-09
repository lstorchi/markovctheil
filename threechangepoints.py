import scipy
import numpy
import math
import sys

import scipy.io

import matplotlib.pyplot as plt

sys.path.append("./module")
import basicutils
import changemod

filename = ""
c_p1 = 0
c_p2 = 0
c_p3 = 0

if len(sys.argv) == 5:
    filename = sys.argv[1]
    c_p1 = int(sys.argv[2])
    c_p2 = int(sys.argv[3])
    c_p3 = int(sys.argv[4])
else:
    print "usage: ", sys.argv[0], " ratingmtx changepoint1 changepoint2 changepoint3"
    exit()
 
msd = scipy.io.loadmat(filename)

namems = "ratings"

if not(namems in msd.keys()):
    print ("Cannot find " + namems + " in " + filename)
    print (msd.keys())
    exit(1)

rm = msd[namems]
countries = rm.shape[0]
rating=numpy.max(rm)
time=rm.shape[1]

errmsg = ""
L1, L2, L3, L4 = changemod.compute_three_cp(rm, c_p1, c_p2, c_p3, errmsg)

#if (L==None): 
#    print errmsg
#    exit(1)

summa = L1 + L2 + L3 + L4 
print  L1, L2, L3, L4, summa 

