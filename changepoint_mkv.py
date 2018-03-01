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
c_p=time/2

if (rating <= 0) or (rating > 8):
    errmsg.append("rating " + rating + " is not a valid value")
    exit(1)

num = numpy.zeros((rating,rating), dtype='int64')
den = numpy.zeros(rating, dtype='int64')
pr = numpy.zeros((rating,rating), dtype='float64')

L = 0.0

for i in range(rating):
    for j in range(rating):
        for t in range(time-1):
            if (rm[rownum, t] == (i+1)) and (rm[rownum, t+1] == (j+1)):
                num[i, j] = num[i, j] + 1
                
    den[i] = sum(num[i])

    if (den[i] > 0.0):
        for j in range(rating):
            val = numpy.float64(num[i,j])/numpy.float64(den[i])
            if (val > 0.0):
                L += num[i,j]*math.log(val) 

for i in range(rating):
    for j in range(rating):
        if den[i] != 0:
           pr[i, j] = float(num[i, j])/float(den[i])
        else: 
           pr[i, j] = 0
           pr[i, i] = 1
   
num1 = numpy.zeros((rating,rating), dtype='int64')
den1 = numpy.zeros(rating, dtype='int64')
pr1 = numpy.zeros((rating,rating),dtype='float64')

L1 = 0.0 

for i in range(rating):
     for j in range(rating):
          for t in range(c_p-1):
             if (rm[rownum, t] == (i+1)) and (rm[rownum, t+1] == (j+1)):
                 num1[i, j] = num1[i, j] + 1
        

     den1[i] = sum(num1[i])

     if (den1[i] > 0.0):
        for j in range(rating):
           val = numpy.float64(num1[i,j])/numpy.float64(den1[i])
           if (val > 0.0):
              L1 += num1[i,j]*math.log(val) 

for i in range(rating):
    for j in range(rating):
        if den1[i] != 0:
           pr1[i, j] = float(num1[i, j])/float(den1[i])
        else: 
           pr1[i, j] = 0        
           pr1[i,i] = 1  


num2 = numpy.zeros((rating,rating), dtype='int64')
den2 = numpy.zeros(rating, dtype='int64')
pr2 = numpy.zeros((rating,rating),dtype='float64')
 
L2 = 0.0 

for i in range(rating):
     for j in range(rating):
         for t in range(c_p,time-1) :
             if (rm[rownum, t] == (i+1)) and (rm[rownum, t+1] == (j+1)):
                 num2[i, j] = num2[i, j] + 1
         
     den2[i] = sum(num2[i])

     if (den2[i] > 0.0):
         for j in range(rating):
            val = numpy.float64(num2[i,j])/numpy.float64(den2[i])
            if (val > 0.0):
                L2 += num2[i,j]*math.log(val) 
 
for i in range(rating):
    for j in range(rating):
        if den2[i] != 0.0:
           pr2[i, j] = float(num2[i, j])/float(den2[i])
        else: 
           pr2[i, j] = 0
           pr2[i,i] = 1

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
