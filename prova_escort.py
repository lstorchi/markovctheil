import numpy.linalg
import scipy.stats
import scipy.io
import argparse
import numpy
import math
import sys
import os

import os.path

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

sys.path.append("./module")
import basicutils

filename1 = ""
filename2 = ""

if len(sys.argv) == 3:
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
else:
    print "usage: ", sys.argv[0], " ratingmtx interestmtx " 
    exit()

msd = scipy.io.loadmat(filename1)
bpd = scipy.io.loadmat(filename2)

namems = "data_ms"
namebp = "nmat"

if not(namems in msd.keys()):
    print ("Cannot find " + namems + " in " + filename1)
    print (msd.keys())
    exit(1)

if not(namebp in bpd.keys()):
    print ("Cannot find " + namebp + " in " + filename2)
    print (bpd.keys())
    exit(1)

if msd[namems].shape[0] != bpd[namebp].shape[0]:
    print ("wrong dim of the input matrix")
    exit(1)

rm = msd[namems]
ir = bpd[namebp]

rm=rm[:,0:-6]
ir=ir[:,0:-6]

countries = rm.shape[0]
rating = numpy.max(rm)
time = rm.shape[1]

for i in range(len(ir)):
       for j in range(len(ir[0])):
           if math.isnan(ir[i, j]):
              ir[i, j] = float('inf')

benchmark = numpy.amin(ir, 0)
r = numpy.zeros((countries,time), dtype='float64')

for i in range(countries):
    for j in range(time):
        r[i, j] = ir[i, j] - benchmark[j]

for i in range(len(ir)):
       for j in range(len(r[0])):
           if (ir[i, j] == float('Inf')):
               ir[i, j] = 0


for i in range(len(r)):
       for j in range(len(r[0])):
           if (r[i, j] == float('Inf')):
               r[i, j] = 0


R = numpy.sum(r, axis=0)
R_i = numpy.sum(ir, axis=0)



rc = numpy.zeros((rating,time), dtype='float64')
ri = numpy.zeros((rating,time), dtype='float64')
s_r = numpy.zeros((rating,time), dtype='float64')
s_i = numpy.zeros((rating,time), dtype='float64')


for t in range(time):
    for j in range(countries):
        for i in range(rating):
            if rm[j,t] == i:
                rc[i,t] = rc[i,t] + r[j,t] #totale credt spread pagati dalla classe di rating i
                ri[i,t] = ri[i,t] + ir[j,t] #totale interest rates  pagati dalla classe di rating i
	
        s_r[i,t] = rc[i,t] / R[t]
	s_i[i,t] = ri[i,t] / R_i[t]

print "Done "

DIM = 11
#DIM = 401

T = numpy.zeros((time,DIM), dtype='float64')
p_s = numpy.zeros((rating,time), dtype='float64')
E_r = numpy.zeros((rating,time), dtype='float64')
r_s = numpy.zeros((time), dtype='float64')
 
s = 0
es = 0.05


for i in range(rating):
    for t in range(time):        
        p_s[i,t] = s_i[i,t]**es

r_s = numpy.sum(p_s, axis=0)
      	    
for s in range(0,DIM):

    for t in range(time):
        for i in range(rating):
	    E_r[i,t] = p_s[i,t]/r_s[t]
            if E_r[i,t] != 0:
                T[t,s] += E_r[i,t] * math.log(float(countries) * E_r[i,t])

    es += 0.05

print(rc.shape)

basicutils.mat_to_file(rc, "rcmtx.txt")

basicutils.mat_to_file(p_s, "psrmtx.txt")


