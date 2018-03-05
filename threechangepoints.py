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

rating=numpy.max(rm)
time=rm.shape[1]

errmsg = ""

fp = open("change.txt", "w")

maxval = -1.0 * float("inf")
cp1 = 0
cp2 = 0
cp3 = 0
for c_p1 in range(1, time-1):
    print c_p1 , " of ", time-2
    for c_p2 in range(c_p1+1, time):
        print "   ", c_p2 , " of ", time-1 
        for c_p3 in range(c_p2+1, time):
            L1, L2, L3, L4 = changemod.compute_three_cp(rm, c_p1, c_p2, c_p3, errmsg)
            
            summa = L1+L2+L3+L4
            if (maxval < summa):
                maxval = summa
                cp1 = c_p1
                cp2 = c_p2
                cp3 = c_p3
            
            fp.write(str(c_p1) + " " + str(c_p2) + " " + str(c_p3) + " " + \
                    str(summa)+ "\n")

fp.close()
print cp1, cp2, cp3, maxval
