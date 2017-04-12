import numpy.linalg
import numpy.random
import scipy.stats
import scipy.io
import numpy
import math
import sys
import os

###############################################################################

def meant (ms):

    gcont = 0.0
    gmt = 0.0
    for i in range(ms.shape[0]):
    
        v1 = ms[i, 0]
        tstart = 0
        
        mt = 0.0
        cont = 0.0
        
        for j in range(1, ms.shape[1]):
            if ms[i,j] != v1:
                tend = j
                v1 = ms[i,j]
                dt = tend-tstart
                mt += float(dt)
                cont += 1.0
                tstart = j
        
        if cont != 0.0:
            print i+1, " ", mt/cont
            gcont += 1.0
            gmt += (mt/cont)

    gmt = gmt / gcont

    return gmt

###############################################################################

filename1 = "ms0.mat"
filename2 = "ms1.mat"
filename3 = "ms2.mat"

if len(sys.argv) != 4:
    print "usage: ", sys.argv[0], \
            " msmatfilename ms1matfilename ms3matfilename" 
    exit(1)
else:
    filename1 = sys.argv[1] 
    filename2 = sys.argv[2]
    filename3 = sys.argv[3]

numpy.random.seed(9001)

ms0d = scipy.io.loadmat(filename1)
ms1d = scipy.io.loadmat(filename2)
ms2d = scipy.io.loadmat(filename3)

ms0 = ms0d["ms"]
ms1 = ms1d["sp"]
ms2 = ms2d["ft"]

print meant(ms0)
print ""
print meant(ms1)
print ""
print meant(ms2)
print ""
