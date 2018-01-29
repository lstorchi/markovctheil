import scipy
import numpy
import math
import sys

import scipy.stats
import scipy.io

import matplotlib.pyplot as plt

sys.path.append("./module")
import basicutils

filename = ""

if len(sys.argv) == 2:
    filename = sys.argv[1]
else:
    print "usage: ", sys.argv[0], " ratingmtx "
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
c_p=time/2

if (rating <= 0) or (rating > 8):
    errmsg.append("rating " + rating + " is not a valid value")
    exit(1)

nk = numpy.zeros((rating,rating,countries), dtype='int64')
num = numpy.zeros((rating,rating), dtype='int64')
den = numpy.zeros(rating, dtype='int64')
pr = numpy.zeros((rating,rating), dtype='float64')

L = 0.0

for i in range(rating):
    for j in range(rating):
        for c  in range(countries):
            for t in range(time-1):
                if (rm[c, t] == (i+1)) and (rm[c, t+1] == (j+1)):
                    nk[i, j, c] = nk[i, j, c] + 1
                
                num[i, j] = sum(nk[i, j])

    den[i] = sum(num[i])

    if (den[i] > 0.0):
        if (num[i,j]/den[i] > 0.0):
            L += num[i,j]*math.log(num[i,j]/den[i]) 

for i in range(rating):
    for j in range(rating):
        if den[i] != 0:
           pr[i, j] = float(num[i, j])/float(den[i])
        else: 
           pr[i, j] = 0
           pr[i,i] = 1
   
nk1 = numpy.zeros((rating,rating,countries), dtype='int64')
num1 = numpy.zeros((rating,rating), dtype='int64')
den1 = numpy.zeros(rating, dtype='int64')
pr1 = numpy.zeros((rating,rating),dtype='float64')

L1 = 0.0 

for i in range(rating):
     for j in range(rating):
        for c in range(countries):
             for t in range(c_p-1):
 
                if (rm[c, t] == (i+1)) and (rm[c, t+1] == (j+1)):
                    nk1[i, j, c] = nk1[i, j, c] + 1
                num1[i, j] = sum(nk1[i, j])
             den1[i] = sum(num1[i])
             if (den1[i] > 0.0):
                if (num1[i,j]/den1[i] > 0.0):
                   L1 += num1[i,j]*math.log(num1[i,j]/den1[i]) 

for i in range(rating):
    for j in range(rating):
        if den1[i] != 0:
           pr1[i, j] = float(num1[i, j])/float(den1[i])
        else: 
           pr1[i, j] = 0        
           pr1[i,i] = 1  



nk2 = numpy.zeros((rating,rating,countries), dtype='int64')
num2 = numpy.zeros((rating,rating), dtype='int64')
den2 = numpy.zeros(rating, dtype='int64')
pr2 = numpy.zeros((rating,rating),dtype='float64')
 
L2 = 0.0 

for i in range(rating):
     for j in range(rating):
         for c in range(countries):
              for t in range(c_p,time-1) :

                  if (rm[c, t] == (i+1)) and (rm[c, t+1] == (j+1)):
                      nk2[i, j, c] = nk2[i, j, c] + 1
                  num2[i, j] = sum(nk2[i, j])
              den2[i] = sum(num2[i])
              if (den2[i] > 0.0):
                  if (num2[i,j]/den2[i] > 0.0):
                      L2 += num2[i,j]*math.log(num2[i,j]/den2[i]) 
 
for i in range(rating):
    for j in range(rating):
        if den1[i] != 0:
           pr2[i, j] = float(num2[i, j])/float(den2[i])
        else: 
           pr2[i, j] = 0
           pr2[i,i] = 1

print L, L1, L2
lamda = -2*((L1+L2)/L)
