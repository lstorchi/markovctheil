import numpy.linalg
import numpy.random
import scipy.stats
import scipy.io
import argparse
import random
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

countries = rm.shape[0]
rating = numpy.max(rm)
time = rm.shape[1]

benchmark = numpy.amin(ir, 0)
r = numpy.zeros((countries,time), dtype='float64')

for i in range(countries):
    for j in range(time):
        r[i, j] = ir[i, j] - benchmark[j]

R = numpy.sum(r, axis=0)
Ri = numpy.sum(ir, axis=0)

rc = numpy.zeros((rating,time), dtype='float64')
ri = numpy.zeros((rating,time), dtype='float64')
s_r = numpy.zeros((rating,time), dtype='float64')
s_i = numpy.zeros((rating,time), dtype='float64')

for i in range(rating):
    for t in range(time):

        for j in range(countries):
            if rm[j,t] == i:
                rc[i,t] = rc[i,t] + r[j,t] #totale credt spread pagati dalla classe di rating i
                ri[i,t] = ri[i,t] + ir[j,t] #totale interest rates  pagati dalla classe di rating i
	
        s_r[i,t] = rc[i,t] / R[t]
	s_i[i,t] = ri[i,t] / Ri[t]

print "Done "

DIM = 11
#DIM = 401

T = numpy.zeros((time,DIM), dtype='float64')
p_s = numpy.zeros((rating,time), dtype='float64')
E_r = numpy.zeros((rating,time), dtype='float64')
 
s = 0
es = 0.05
for s in range(0,DIM):

    for i in range(rating):
        for t in range(time):        
            p_s[i,t] = s_i[i,t]**es
            r_s = numpy.sum(p_s, axis=0)
      	    E_r[i,t] = p_s[i,t]/r_s[t]

    for t in range(time):
        for i in range(rating):
            if E_r[i,t] != 0:
                T[t,s] += E_r[i,t] * math.log(float(countries) * E_r[i,t])

    es += 0.05

basicutils.mat_to_file(T, "tmtx.txt")

print es
