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

import mainmkvcmp

sys.path.append("./module")
import basicutils

parser = argparse.ArgumentParser()

parser.add_argument("-m","--rmat-filename", help="Rating mattrix filename", \
        type=str, required=True, dest="rmatfilename")
parser.add_argument("-b", "--imat-filename", help="Interest rates matrix filename", \
        type=str, required=True, dest="imatfilename")
parser.add_argument("-s", "--step", help="Step ", \
        type=float, required=False, default=0.25, dest="step")
parser.add_argument("-t", "--time-prev", help="Time prev ", \
        type=int, required=False, default=37, dest="tprev")
parser.add_argument("-n", "--max-run", help="Num. of run required ", \
        type=int, required=True, dest="maxrun")
parser.add_argument("-M", "--name-of-matrix", help="Name of MS matrix ", \
        type=str, required=False, default="ms", dest="nameofmatrix")
parser.add_argument("-B", "--name-of-bpmatrix", help="Name of BP matrix ", \
        type=str, required=False, default="i_r", dest="nameofbpmatrix")
parser.add_argument("-v", "--verbose", help="increase output verbosity", \
        default=False, action="store_true")
parser.add_argument("-i", "--time-inf", help="Simulate infinie time", \
        default=False, action="store_true", dest="timeinf")
parser.add_argument("-S", "--seed", help="using a seed for the random generator", \
        default=False, action="store_true", dest="seed")

if len(sys.argv) == 1:
    parser.print_help()
    exit(1)

args = parser.parse_args()

namebp = args.nameofbpmatrix
timeinf = args.timeinf
verbose = args.verbose
filename1 = args.rmatfilename
filename2 = args.imatfilename
step = args.step
tprev = args.tprev
numofrun = args.maxrun
namems = args.nameofmatrix

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

<<<<<<< HEAD
if rating > 2:
    a = y[2][:Nn[2]]
    a = a[numpy.isfinite(a)]
    Pa, Ta, ha, xa, nbins = get_histo(a, step)

    plt.hist(a, normed=False, bins=nbins, facecolor='green')
    plt.xlabel("bp")
    plt.ylabel("f(x)")
    plt.title("A")
    plt.grid(True)
    plt.savefig("a_"+str(run)+".eps")

    histo_to_file (xa, ha, "a_"+str(run)+".txt")

    allratings.append(a)
    Mean.append(numpy.mean(a))
    Ti.append(Ta)

    print "A done"

if rating > 3: 
    bbb = y[3][:Nn[3]]
    bbb = bbb[numpy.isfinite(bbb)]
    Pbbb, Tbbb, hbbb, xbbb, nbins = get_histo(bbb, step)

    plt.hist(bbb, normed=False, bins=nbins, facecolor='green')
    plt.xlabel("bp")
    plt.ylabel("f(x)")
    plt.title("BBB")
    plt.grid(True)
    plt.savefig("bbb_"+str(run)+".eps")

    histo_to_file (xbbb, hbbb, "bbb_"+str(run)+".txt")

    allratings.append(bbb)
    Mean.append(numpy.mean(bbb))
    Ti.append(Tbbb)

    print "BBB done"

if rating > 4:
    bb = y[4][:Nn[4]]
    bb = bb[numpy.isfinite(bb)]
    Pbb, Tbb, hbb, xbb, nbins = get_histo(bb, step)

    plt.hist(bb, normed=False, bins=nbins, facecolor='green')
    plt.xlabel("bp")
    plt.ylabel("f(x)")
    plt.title("BB")
    plt.grid(True)
    plt.savefig("bb_"+str(run)+".eps")

    histo_to_file (xbb, hbb, "bb_"+str(run)+".txt")

    allratings.append(bb)
    Mean.append(numpy.mean(bb))
    Ti.append(Tbb)

    print "BB done"

if rating > 5:
    b = y[5][:Nn[5]]
    b = b[numpy.isfinite(b)]
    Pb, Tb, hb, xb, nbins = get_histo(b, step)

    plt.hist(b, normed=False, bins=nbins, facecolor='green')
    plt.xlabel("bp")
    plt.ylabel("f(x)")
    plt.title("B")
    plt.grid(True)
    plt.savefig("b_"+str(run)+".eps")

    histo_to_file (xb, hb, "b_"+str(run)+".txt")

    allratings.append(b)
    Mean.append(numpy.mean(b))
    Ti.append(Tb)

    print "B done"

if rating > 6:
    cc = y[6][:Nn[6]]
    cc = cc[numpy.isfinite(cc)]
    Pcc, Tcc, hcc, xcc, nbins = get_histo(cc, step)

    plt.hist(cc, normed=False, bins=nbins, facecolor='green')
    plt.xlabel("bp")
    plt.ylabel("f(x)")
    plt.title("CC")
    plt.grid(True)
    plt.savefig("cc_"+str(run)+".eps")

    histo_to_file (xcc, hcc, "cc_"+str(run)+".txt")

    allratings.append(cc)
    Mean.append(numpy.mean(cc))
    Ti.append(Tcc)

    print "CC done"

if rating > 7:
    d = y[rating-1][:Nn[7]]
    allratings.append(d)
    Pd, Td, hd, xd, nbins = get_histo(d, step)

    plt.hist(d, normed=False, bins=nbins, facecolor='green')
    plt.xlabel("bp")
    plt.ylabel("f(x)")
    plt.title("D")
    plt.grid(True)
    plt.savefig("d_"+str(run)+".eps")
    
    histo_to_file (xd, hd, "d_"+str(run)+".txt")

    allratings.append(d)
    Mean.append(numpy.mean(d))
    Ti.append(Td)

    print "D done"

fval = 0.0
pval = 0.0

if rating == 1:
    fval, pval = scipy.stats.f_oneway (allratings[0])
elif rating == 2:
    fval, pval = scipy.stats.f_oneway (allratings[0], allratings[1])
elif rating == 3:
    fval, pval = scipy.stats.f_oneway (allratings[0], allratings[1], \
            allratings[2])
elif rating == 4: 
    fval, pval = scipy.stats.f_oneway (allratings[0], allratings[1], \
            allratings[2], allratings[3])
elif rating == 5:
    fval, pval = scipy.stats.f_oneway (allratings[0], allratings[1], \
            allratings[2], allratings[3], allratings[4])
elif rating == 6:
    fval, pval = scipy.stats.f_oneway (allratings[0], allratings[1], \
            allratings[2], allratings[3], allratings[4], allratings[5])
elif rating == 7:
    fval, pval = scipy.stats.f_oneway (allratings[0], allratings[1], \
            allratings[2], allratings[3], allratings[4], allratings[5], \
            allratings[6])
elif rating == 8:
    fval, pval = scipy.stats.f_oneway (allratings[0], allratings[1], \
            allratings[2], allratings[3], allratings[4], allratings[5], \
            allratings[6], allratings[7])
print " "

oufilename = "1wayanova_"+str(run)+".txt"

if os.path.exists(oufilename):
    os.remove(oufilename)

outf = open(oufilename, "w")

outf.write("F-value: %f\n"%fval)
outf.write("P value: %f\n"%pval)

outf.close()

s_t = numpy.zeros((countries,time), dtype='float64')

for i in range(r.shape[0]):
    for j in range(r.shape[1]):
        if math.isnan(r[i][j]):
            r[i][j] = 0.0

R_t = numpy.sum(r, axis=0)
T_t = numpy.zeros(time, dtype='float64')

for t in range(time):
    for k in range(countries):
        s_t[k][t] = r[k][t] / R_t[t]
        if s_t[k][t] != 0:
            T_t[t] += s_t[k][t]*math.log(float(countries) * s_t[k][t])
<<<<<<< HEAD
#print "entropia storica", T_t
oufilename = "T_t.txt"
=======

#print "entropia storica", T_t
oufilename = "entropy_histi_"+str(run)+".txt"
vct_to_file(T_t, oufilename)

>>>>>>> ec721321876ed950c52adefb17f5194ae1fae182
X = numpy.random.rand(countries,tprev,run)
cdf = numpy.zeros((rating,rating), dtype='float64')

x = numpy.zeros((countries,tprev,run), dtype='int')
bp = numpy.zeros((countries,tprev,run), dtype='float64')
xm = numpy.zeros((countries,tprev), dtype='float64')
r_prev = numpy.zeros((tprev,run), dtype='float64')
Var = numpy.zeros((tprev), dtype='float64')
tot = numpy.zeros((rating,tprev,run), dtype='float64')
cont = numpy.zeros((rating,tprev,run), dtype='int')
ac = numpy.zeros((rating,tprev,run), dtype='float64')
t1 = numpy.zeros((tprev,run), dtype='float64')
t2 = numpy.zeros((tprev,run), dtype='float64')
term = numpy.zeros((tprev,run), dtype='float64')
entr = numpy.zeros((tprev,run), dtype='float64')
=======
>>>>>>> d894f1a449c31edbd260f0d6b2fdee57ec2ef5e9
entropia = numpy.zeros(tprev, dtype='float64')
var = numpy.zeros((tprev), dtype='float64')

rating = numpy.max(ms)

pr = numpy.zeros((rating,rating), dtype='float64')
meanval = []
stdeval = []
 
allratings = []
allratingsnins = []

if not mainmkvcmp.main_mkc_comp (ms, i_r, timeinf, step, tprev, \
        numofrun, verbose, True, args.seed, errmsg, entropia, \
        var, allratings, allratingsnins, pr, meanval, stdeval):
    for m in errmsg:
        print m
    exit(1)
