import numpy.linalg
import numpy.random
import scipy.stats
import scipy.io
import numpy
import math
import sys
import os

###############################################################################

def mat_to_stdout (mat):

    for i in range(mat.shape[0]):
       for j in range(mat.shape[1]):
           sys.stdout.write ("%f "%mat[i][j])
       sys.stdout.write ("\n")

###############################################################################

def mat_to_file (bpm, oufilename):

    if os.path.exists(oufilename):
        os.remove(oufilename)
    
    outf = open(oufilename, "w")

    for i in range(bpm.shape[0]):
        for j in range(bpm.shape[1]):
            outf.write(" %f "%(bpm[i][j]))
        outf.write("\n")
        
    outf.close()

###############################################################################

def histo_to_file (xh, h, oufilename):

    if os.path.exists(oufilename):
        os.remove(oufilename)
    
    outf = open(oufilename, "w")

    for i in range(len(h)):
        outf.write("%f %f\n"%((xh[i]+xh[i+1])/2.0, h[i]))
        
    outf.close()

###############################################################################

def progress_bar (count, total, status=''):

    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() 

###############################################################################

def get_histo (v, step):
    
    minh = min(v)
    maxh = max(v)
    nbins = int ((maxh - minh) / step)
    h, xh = numpy.histogram(v, bins=nbins, range=(minh, maxh))
    p = h/float(sum(h))

    t = sum((p*(math.log(len(p)))*p))

    return p, t, h, xh

###############################################################################

filename1 = "ms.mat"
filename2 = "bp.mat"
step = 0.25 
run = 100000
tprev = 37 # mesi previsione
namems = 'ms'
namebp = 'i_r'

if len(sys.argv) != 6:
    print "usage: ", sys.argv[0], " msmatfilename bpmatfilename step tprev run" 
    exit(1)
else:
    filename1 = sys.argv[1] 
    filename2 = sys.argv[2]
    step = float(sys.argv[3])
    tprev = int(sys.argv[4])
    run = int(sys.argv[5])

#numpy.random.seed(9001)

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

Nk = numpy.zeros((rating,rating,countries), dtype='int64')
Num = numpy.zeros((rating,rating), dtype='int64')
Den = numpy.zeros(rating, dtype='int64')
Pr = numpy.zeros((rating,rating), dtype='float64')

verbose = False

for k in range(countries):
    for t in range(time-1):
        for i in range(rating):
            for j in range(rating):
                if (ms[k][t] == (i+1)) and (ms[k][t+1] == (j+1)):
                    Nk[i][j][k] = Nk[i][j][k] + 1

                Num[i][j] = sum(Nk[i][j])

            Den[i] = sum(Num[i])

    progress_bar(k+1, countries)

#print Num 
#print Den

for i in range(rating):
    for j in range(rating):
        if Den[i] != 0:
            Pr[i][j] = float(Num[i][j])/float(Den[i])
        else: 
            Pr[i][j] = 0.0

newPr = Pr - numpy.identity(rating, dtype='float64')

#print newPr

s, v, d = numpy.linalg.svd(newPr)
#print numpy.mean(v)

for i in range(len(i_r)):
    for j in range(len(i_r[0])):
        if math.isnan(i_r[i][j]):
           i_r[i][j] = float('inf')

benchmark = numpy.amin(i_r, 0)

r = numpy.zeros((countries,time), dtype='float64') 

for k in range(countries):
    for Time in range(time):
        r[k][Time] = i_r[k][Time] - benchmark[Time]

for i in range(len(r)):
    for j in range(len(r[0])):
        if (r[i][j] == float('Inf')):
           r[i][j] = float('nan')

ist = numpy.zeros((rating,time*countries), dtype='float64')
Nn = numpy.zeros((rating), dtype='int')

for i in range(rating):
    for k in range(countries):
        for Time in range(time):
            if ms[k][Time] == i+1: 
                Nn[i] = Nn[i] + 1 
                ist[i][Nn[i]-1] = r[k][Time]

Mean = numpy.zeros((rating), dtype='float64')
y = numpy.zeros((ist.shape[0], Nn[0]), dtype='float64')
for i in range(len(ist)):
    y[i] = ist[i][0:Nn[0]]

allratings = []
Mean = []
Ti = []

if rating > 0:
    aaa = y[0][:Nn[0]]
    aaa = aaa[numpy.isfinite(aaa)]
    Paaa, Taaa, aaa, xaaa = get_histo(aaa, step)

    histo_to_file (xaaa, aaa, "aaa_"+str(run)+".txt")

    allratings.append(aaa)
    Mean.append(numpy.mean(aaa))
    Ti.append(Taaa)

if rating > 1:
    aa = y[1][:Nn[1]]
    aa = aa[numpy.isfinite(aa)]
    Paa, Taa, aa, xaa = get_histo(aa, step)

    histo_to_file (xaa, aa, "aa_"+str(run)+".txt")

    allratings.append(aa)
    Mean.append(numpy.mean(aa))
    Ti.append(Taa)

if rating > 2:
    a = y[2][:Nn[2]]
    a = a[numpy.isfinite(a)]
    Pa, Ta, a, xa = get_histo(a, step)

    histo_to_file (xa, a, "a_"+str(run)+".txt")

    allratings.append(a)
    Mean.append(numpy.mean(a))
    Ti.append(Ta)

if rating > 3: 
    bbb = y[3][:Nn[3]]
    bbb = bbb[numpy.isfinite(bbb)]
    Pbbb, Tbbb, bbb, xbbb = get_histo(bbb, step)

    histo_to_file (xbbb, bbb, "bbb_"+str(run)+".txt")

    allratings.append(bbb)
    Mean.append(numpy.mean(bbb))
    Ti.append(Tbbb)

if rating > 4:
    bb = y[4][:Nn[4]]
    bb = bb[numpy.isfinite(bb)]
    Pbb, Tbb, bb, xbb = get_histo(bb, step)

    histo_to_file (xbb, bb, "bb_"+str(run)+".txt")

    allratings.append(bb)
    Mean.append(numpy.mean(bb))
    Ti.append(Tbb)

if rating > 5:
    b = y[5][:Nn[5]]
    b = b[numpy.isfinite(b)]
    Pb, Tb, b, xb = get_histo(b, step)

    histo_to_file (xb, b, "b_"+str(run)+".txt")

    allratings.append(b)
    Mean.append(numpy.mean(b))
    Ti.append(Tb)

if rating > 6:
    cc = y[6][:Nn[6]]
    cc = cc[numpy.isfinite(cc)]
    Pcc, Tcc, cc, xcc = get_histo(cc, step)

    histo_to_file (xcc, cc, "cc_"+str(run)+".txt")

    allratings.append(cc)
    Mean.append(numpy.mean(cc))
    Ti.append(Tcc)

if rating > 7:
    d = y[rating-1][:Nn[7]]
    allratings.append(d)
    Pd, Td, d, xd = get_histo(d, step)
    
    histo_to_file (xd, d, "d_"+str(run)+".txt")

    allratings.append(d)
    Mean.append(numpy.mean(d))
    Ti.append(Td)

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
entropia = numpy.zeros(tprev, dtype='float64')
R_prev = numpy.zeros(tprev, dtype='float64')

for j in range(run):

    # da controllare
    for i in range(countries):
        x[i][0][j] = ms[i][time-1]

    for i in range (rating):
        cdf[i][0] = Pr[i][0]

    for i in range(rating):
        for k in range(1,rating):
            cdf[i][k] = Pr[i][k] + cdf[i][k-1]

    for c in range(countries):
        if X[c][0][j] <= cdf[x[c][0][j]-1][0]:
            x[c][1][j] = 1

        for k in range(1,rating):
            if (cdf[x[c][0][j]-1][k-1] < X[c][0][j]) and \
                    (X[c][0][j] <= cdf[x[c][0][j]-1][k] ):
               x[c][1][j] = k + 1

        for t in range(2,tprev):
            if X[c][t-1][j] <= cdf[x[c][t-1][j]-1][0]:
                x[c][t][j] = 1

            for k in range(1,rating):
                if (cdf[x[c][t-1][j]-1][k-1] < X[c][t-1][j]) \
                        and (X[c][t-1][j] <= cdf[x[c][t-1][j]-1][k]):
                  x[c][t][j] = k + 1

    for t in range(tprev):
        for c in range(countries):
            for i in range(rating):
                if x[c][t][j] == i+1:
                    bp[c][t][j] = Mean[i]
                    cont[i][t][j] = cont[i][t][j] + 1
                    tot[i][t][j] = cont[i][t][j] * Mean[i]
            
        summa = 0.0
        for a in range(bp.shape[0]):
            summa += bp[a][t][j]
        r_prev[t][j] = summa

    for t in range(tprev):
        for i in range(rating):
             ac[i][t][j] = tot[i][t][j]/r_prev[t][j]
             if ac[i][t][j] != 0.0:
                 t1[t][j] += (ac[i][t][j]*Ti[i])
                 t2[t][j] += (ac[i][t][j]*math.log(float(rating)*ac[i][t][j]))
                 if cont[i][t][j] != 0:
                    term[t][j] += ac[i][t][j]* \
                            math.log(float(countries)/(float(rating)*cont[i][t][j]))
 
        entr[t][j] = t1[t][j] + t2[t][j] + term[t][j]
        R_prev[t] = numpy.mean(r_prev[t][j])

    progress_bar(j+1, run)

print " "

oufilename = "entropy_"+str(run)+".txt"

if os.path.exists(oufilename):
    os.remove(oufilename)

outf = open(oufilename, "w")

for t in range(tprev):
    entropia[t] =numpy.mean(entr[t])
    Var[t] = numpy.std(entr[t])

for t in range(tprev):
    outf.write("%d %f %f \n"%(t+1, entropia[t], Var[t]))

outf.close()

acm = numpy.zeros((rating,tprev), dtype='float64')
for i in range(acm.shape[0]):
    for j in range(acm.shape[1]):
        acm[i][j] = numpy.mean(ac[i][j])

oufilename = "acm_"+str(run)+".txt"

mat_to_file (acm, oufilename)

bpm = numpy.zeros((countries,tprev), dtype='float64')
for i in range(bpm.shape[0]):
    for j in range(bpm.shape[1]):
        bpm[i][j] = numpy.mean(bp[i][j])

oufilename = "bpm_"+str(run)+".txt"

mat_to_file (bpm, oufilename)

