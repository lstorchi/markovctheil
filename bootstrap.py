import scipy
import numpy
import math
import sys

import scipy.stats
import scipy.io

import matplotlib.pyplot as plt

sys.path.append("./module")
import mainmkvcmp
import basicutils
import changemod

filename = ""
c_p = 0 
numofrun = 0

if len(sys.argv) == 4:
    filename = sys.argv[1]
    c_p = int(sys.argv[2])
    numofrun = int(sys.argv[3])
else:
    print "usage: ", sys.argv[0], " ratingmtx changepoint numofrun"
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

lamda = 2.0*((L1+L2)-L)

print lamda

for i in range(numofrun):
    x = mainmkvcmp.main_mkc_prop (rm, pr1)
    L, L1, L2, prnull = changemod.compute_ls(x, c_p, errmsg)

    lamda = 2.0*((L1+L2)-L)

    print i+1 , " ==> ", lamda 

