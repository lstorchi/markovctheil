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

sys.path.append("./module")
import basicutils

parser = argparse.ArgumentParser()

parser.add_argument("-m","--msmat-filename", help="MS mat filename", \
        type=str, required=True, dest="msmatfilename")
parser.add_argument("-b", "--bpmat-filename", help="BP mat filename", \
        type=str, required=True, dest="bpmatfilename")
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
filename1 = args.msmatfilename
filename2 = args.bpmatfilename
step = args.step
tprev = args.tprev
numofrun = args.maxrun
namems = args.nameofmatrix

if args.seed:
  numpy.random.seed(9001)

if not (os.path.isfile(filename1)):
    print "File ", filename1, " does not exist "
    exit(1)

if not (os.path.isfile(filename2)):
    print "File ", filename2, " does not exist "
    exit(1)

msd = scipy.io.loadmat(filename1)
bpd = scipy.io.loadmat(filename2)

if not(namems in msd.keys()):
    print "Cannot find ", namems, " in ", filename1
    print msd.keys()
    exit(1)

if not(namebp in bpd.keys()):
    print "Cannot find ", namebp, " in ", filename2
    print bpd.keys()
    exit(1)

if msd[namems].shape[0] != bpd[namebp].shape[0]:
    print "wrong dim of the input matrix"
    exit(1)

countries = msd[namems].shape[0]
rating = numpy.max(msd[namems])

if (rating <= 0) or (rating > 8):
    print "rating ", rating, " is not a valid value"
    exit(1)

ms = msd[namems]
i_r = bpd[namebp]
time = len(ms[1,:])

pr = numpy.zeros((rating,rating), dtype='float64')
nk = numpy.zeros((rating,rating,countries), dtype='int64')
num = numpy.zeros((rating,rating), dtype='int64')
den = numpy.zeros(rating, dtype='int64')

for k in range(countries):
    for t in range(time-1):
        for i in range(rating):
            for j in range(rating):
                if (ms[k, t] == (i+1)) and (ms[k, t+1] == (j+1)):
                    nk[i, j, k] = nk[i, j, k] + 1

                num[i, j] = sum(nk[i, j])

            den[i] = sum(num[i])

    basicutils.progress_bar(k+1, countries)

for i in range(rating):
    for j in range(rating):
        if den[i] != 0:
            pr[i, j] = float(num[i, j])/float(den[i])
        else: 
            pr[i, j] = 0.0

if timeinf: # matrice delle probabilita' diventa stazionaria tempo elevato 
    print ""
    print "Solve ..."
    ai = numpy.identity(rating, dtype='float64') - numpy.matrix.transpose(pr)
    a = numpy.zeros((rating+1,rating), dtype='float64')

    for i in range(rating):
        for j in range(rating):
            a[i, j] = ai[i, j]

    for i in range(rating):
        a[rating, i] = 1.0 

    b = numpy.zeros(rating+1, dtype='float64')
    b[rating] = 1.0
    x = numpy.linalg.lstsq(a, b)
    for j in range(rating):
        for i in range(rating):
            pr[i, j] = x[0, j] 
 
print " "
if verbose:
  print "Solve SVD "

npr = pr - numpy.identity(rating, dtype='float64')
s, v, d = numpy.linalg.svd(npr)

if verbose:
    print " "
    print "mean value: ", numpy.mean(v)

for i in range(len(i_r)):
    for j in range(len(i_r[0])):
        if math.isnan(i_r[i, j]):
           i_r[i, j] = float('inf')

benchmark = numpy.amin(i_r, 0)

r = numpy.zeros((countries,time), dtype='float64') 

for i in range(countries):
    for j in range(time):
        r[i, j] = i_r[i, j] - benchmark[j]

for i in range(len(r)):
    for j in range(len(r[0])):
        if (r[i, j] == float('Inf')):
           r[i, j] = float('nan')

ist = numpy.zeros((rating,time*countries), dtype='float64')
nn = numpy.zeros((rating), dtype='int')

for i in range(rating):
    for j in range(countries):
        for k in range(time):
            if ms[j, k] == i+1: 
                nn[i] = nn[i] + 1 
                ist[i, nn[i]-1] = r[j, k]

meanval = numpy.zeros((rating), dtype='float64')
y = numpy.zeros((ist.shape[0], nn[0]), dtype='float64')
for i in range(len(ist)):
    y[i] = ist[i, 0:nn[0]]

allratings = []
meanval = []
tiv = []

if rating > 0:
    a, b, c = basicutils.extract_ti_mean (y[0, :nn[0]], step, 0, numofrun, \
            "aaa")
    allratings.append(a)
    meanval.append(b)
    tiv.append(c)

if rating > 1:
    a, b, c = basicutils.extract_ti_mean (y[1, :nn[1]], step, 1, numofrun, \
            "aa")
    allratings.append(a)
    meanval.append(b)
    tiv.append(c)

if rating > 2:
    a, b, c = basicutils.extract_ti_mean (y[2, :nn[2]], step, 2, numofrun, \
            "a")
    allratings.append(a)
    meanval.append(b)
    tiv.append(c)

if rating > 3: 
    a, b, c = basicutils.extract_ti_mean (y[3, :nn[3]], step, 3, numofrun, \
            "bbb")
    allratings.append(a)
    meanval.append(b)
    tiv.append(c)

if rating > 4:
    a, b, c = basicutils.extract_ti_mean (y[4, :nn[4]], step, 4, numofrun, \
            "bb")
    allratings.append(a)
    meanval.append(b)
    tiv.append(c)

if rating > 5:
    a, b, c = basicutils.extract_ti_mean (y[5, :nn[5]], step, 5, numofrun, \
            "b")
    allratings.append(a)
    meanval.append(b)
    tiv.append(c)

if rating > 6:
    a, b, c = basicutils.extract_ti_mean (y[6, :nn[6]], step, 6, numofrun, \
            "cc")
    allratings.append(a)
    meanval.append(b)
    tiv.append(c)

if rating > 7:
    a, b, c = basicutils.extract_ti_mean (y[7, :nn[7]], step, 7, numofrun, \
            "d")
    allratings.append(a)
    meanval.append(b)
    tiv.append(c)

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

oufilename = "1wayanova_"+str(numofrun)+".txt"

if os.path.exists(oufilename):
    os.remove(oufilename)

outf = open(oufilename, "w")

outf.write("F-value: %f\n"%fval)
outf.write("P value: %f\n"%pval)

outf.close()

s_t = numpy.zeros((countries,time), dtype='float64')

for i in range(r.shape[0]):
    for j in range(r.shape[1]):
        if math.isnan(r[i, j]):
            r[i, j] = 0.0

R_t = numpy.sum(r, axis=0)
T_t = numpy.zeros(time, dtype='float64')

for t in range(time):
    for k in range(countries):
        s_t[k, t] = r[k, t] / R_t[t]
        if s_t[k, t] != 0:
            T_t[t] += s_t[k, t]*math.log(float(countries) * s_t[k, t])

#print "entropia storica", T_t
oufilename = "entropy_histi_"+str(numofrun)+".txt"
basicutils.vct_to_file(T_t, oufilename)

bp = numpy.zeros((countries,tprev,numofrun), dtype='float64')
tot = numpy.zeros((rating,tprev,numofrun), dtype='float64')
ac = numpy.zeros((rating,tprev,numofrun), dtype='float64')
xm = numpy.zeros((countries,tprev), dtype='float64')
cdf = numpy.zeros((rating,rating), dtype='float64')
x = numpy.zeros((countries,tprev,numofrun), dtype='int')
cont = numpy.zeros((rating,tprev,numofrun), dtype='int')
r_prev = numpy.zeros((tprev,numofrun), dtype='float64')
term = numpy.zeros((tprev,numofrun), dtype='float64')
entr = numpy.zeros((tprev,numofrun), dtype='float64')
t1 = numpy.zeros((tprev,numofrun), dtype='float64')
t2 = numpy.zeros((tprev,numofrun), dtype='float64')
entropia = numpy.zeros(tprev, dtype='float64')
xi = numpy.random.rand(countries,tprev,numofrun)
var = numpy.zeros((tprev), dtype='float64')

for i in range (rating):
    cdf[i, 0] = pr[i, 0]

for i in range(rating):
    for j in range(1,rating):
        cdf[i, j] = pr[i, j] + cdf[i, j-1]

for run in range(numofrun):

    for c in range(countries):
        x[c, 0, run] = ms[c, time-1]

    for c in range(countries):
        if xi[c, 0, run] <= cdf[x[c, 0, run]-1, 0]:
            x[c, 1, run] = 1

        for k in range(1,rating):
            if (cdf[x[c, 0, run]-1, k-1] < xi[c, 0, run]) and \
                    (xi[c, 0, run] <= cdf[x[c, 0, run]-1, k] ):
               x[c, 1, run] = k + 1

        for t in range(2,tprev):
            if xi[c, t-1, run] <= cdf[x[c, t-1, run]-1, 0]:
                x[c, t, run] = 1

            for k in range(1,rating):
                if (cdf[x[c, t-1, run]-1, k-1] < xi[c, t-1, run]) \
                        and (xi[c, t-1, run] <= cdf[x[c, t-1, run]-1, k]):
                  x[c, t, run] = k + 1

    for t in range(tprev):
        for c in range(countries):
            for i in range(rating):
                if x[c, t, run] == i+1:
                    bp[c, t, run] = meanval[i]
                    cont[i, t, run] = cont[i, t, run] + 1
                    tot[i, t, run] = cont[i, t, run] * meanval[i]
            
        summa = 0.0
        for a in range(bp.shape[0]):
            summa += bp[a, t, run]
        r_prev[t, run] = summa

    for t in range(tprev):
        for i in range(rating):
             ac[i, t, run] = tot[i, t, run]/r_prev[t, run]
             if ac[i, t, run] != 0.0:
                 t1[t, run] += (ac[i, t, run]*tiv[i])
                 t2[t, run] += (ac[i, t, run]*math.log(float(rating)*ac[i, t, run]))
                 if cont[i, t, run] != 0:
                    term[t, run] += ac[i, t, run]* \
                            math.log(float(countries)/(float(rating)*cont[i, t, run]))
 
        entr[t, run] = t1[t, run] + t2[t, run] + term[t, run]

    basicutils.progress_bar(run+1, numofrun)

print " "

oufilename = "entropy_"+str(numofrun)+".txt"

if os.path.exists(oufilename):
    os.remove(oufilename)

outf = open(oufilename, "w")

for t in range(tprev):
    entropia[t] =numpy.mean(entr[t])
    var[t] = numpy.std(entr[t])

for t in range(tprev):
    outf.write("%d %f %f \n"%(t+1, entropia[t], var[t]))

outf.close()

acm = numpy.zeros((rating,tprev), dtype='float64')
for i in range(acm.shape[0]):
    for j in range(acm.shape[1]):
        acm[i, j] = numpy.mean(ac[i, j])

oufilename = "acm_"+str(numofrun)+".txt"

basicutils.mat_to_file (acm, oufilename)

bpm = numpy.zeros((countries,tprev), dtype='float64')
for i in range(bpm.shape[0]):
    for j in range(bpm.shape[1]):
        bpm[i, j] = numpy.mean(bp[i, j])

oufilename = "bpm_"+str(numofrun)+".txt"

basicutils.mat_to_file (bpm, oufilename)
