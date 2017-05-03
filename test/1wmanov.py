import numpy.linalg
import numpy.random
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

sys.path.append("../")
import mainmkvcmp

sys.path.append("../module")
import basicutils

if len(sys.argv) < 3:
    print "Usage: ", sys.argv[0], " N file1 file2 file3 ... "
    exit(1)

N = int(sys.argv[1])


ar = []
am = []
minrat = 1000

for i in range(2, 2+N):
    namebp = "interest_rates"
    timeinf = False
    verbose = True
    filename1 = sys.argv[i]
    filename2 = sys.argv[i]
    step = 0.25
    tprev = 37
    numofrun = 10
    namems = "ratings"

    errmsg = []

    if not (os.path.isfile(filename1)):
        errmsg.append("File " + filename1 + " does not exist ")
        exit(1)

    if not (os.path.isfile(filename2)):
        errmsg.append("File ", filename2, " does not exist ")
        exit(1)
    
    msd = scipy.io.loadmat(filename1)
    bpd = scipy.io.loadmat(filename2)
    
    if not(namems in msd.keys()):
        print "Cannot find " + namems + " in " + filename1
        print msd.keys()
        exit(1)
      
    if not(namebp in bpd.keys()):
        print "Cannot find " + namebp + " in " + filename2
        print bpd.keys()
        exit(1)
        
    if msd[namems].shape[0] != bpd[namebp].shape[0]:
        print "wrong dim of the input matrix"
        exit(1)
    
    ms = msd[namems]
    i_r = bpd[namebp]
    
    entropia = numpy.zeros(tprev, dtype='float64')
    var = numpy.zeros((tprev), dtype='float64')

    rating = numpy.max(ms)

    pr = numpy.zeros((rating,rating), dtype='float64')

    meanval = []
    stdeval = []
 
    allratings = []
    allratingsnins = []
    
    if not mainmkvcmp.main_mkc_comp (ms, i_r, timeinf, step, tprev, \
            numofrun, verbose, True, False, errmsg, entropia, \
            var, allratings, allratingsnins, pr, meanval, stdeval):
        for m in errmsg:
            print m
        exit(1)

    if pr.shape[0] < minrat:
        minrat = pr.shape[0]

    ar.append(allratings)
    am.append(meanval)

if N == 3:
    for j in range(0, minrat):
        fval, pval = scipy.stats.f_oneway (ar[0][j], ar[1][j], ar[2][j])

        print fval, pval


    print "Mean: "
    fval, pval = scipy.stats.f_oneway (am[0], am[1], am[2])
    print fval, pval


