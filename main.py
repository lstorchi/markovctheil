import numpy.linalg
import numpy.random
import scipy.stats
import scipy.io
import numpy
import math
import sys

import sys

filename1 = ""
filename2 = ""
step = 0.25 

'''
if len(sys.argv) != 2:
    print "usage: ", sys.argv[0], " filein" 
    exit(1)
else:
    filename1 = sys.argv[1] 
    filename2 = sys.argv[2]
'''

numpy.random.seed(9001)

msd = scipy.io.loadmat("ms.mat")
bpd = scipy.io.loadmat("bp.mat")

if msd['ms'].shape[0] != bpd['i_r'].shape[0]:
    print "wrong dim of the input matrix"
    exit(1)

countries = msd['ms'].shape[0]
rating = numpy.max(msd['ms'])

ms = msd['ms']
i_r = bpd['i_r']
time = len(ms[1,:])
tprev = 37 # mesi previsione

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

    #print k , " of ", countries
    sys.stdout.flush()

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

#print r

ist = numpy.zeros((rating,time*countries), dtype='float64')
Nn = numpy.zeros((rating), dtype='int')

for i in range(rating):
    for k in range(countries):
        for Time in range(time):
            if ms[k][Time] == i+1: 
                Nn[i] = Nn[i] + 1 
                ist[i][Nn[i]-1] = r[k][Time]

#print Nn
#print ist.shape

#Hfile = scipy.io.loadmat("ist.mat")
#ist1 = Hfile['ist']

#for i in range(rating):
#    for k in range(countries*time):
#        if (ist[i][k] != ist1[i][k]):
#            sys.stdout.write("ist: %f ist1: %f \n"%(ist[i][k], ist1[i][k]))

Mean = numpy.zeros((rating), dtype='float64')
y = numpy.zeros((ist.shape[0], Nn[0]), dtype='float64')
for i in range(len(ist)):
    y[i] = ist[i][0:Nn[0]]

#Yfile = scipy.io.loadmat("y.mat")
#y = Yfile['y']

aaa = y[0][:Nn[0]]
aaa = aaa[numpy.isfinite(aaa)]

aa = []
a = []
bbb = []
bb = []
b = []
cc = [] 
d = []

if rating > 1:
  aa = y[1][:Nn[1]]
  aa = aa[numpy.isfinite(aa)]

if rating > 2:
 a = y[2][:Nn[2]]
 a = a[numpy.isfinite(a)]

if rating > 3: 
  bbb = y[3][:Nn[3]]
  bbb = bbb[numpy.isfinite(bbb)]

if rating > 4:
  bb = y[4][:Nn[4]]
  bb = bb[numpy.isfinite(bb)]

if rating > 5:
  b = y[5][:Nn[5]]
  b = b[numpy.isfinite(b)]

if rating > 6:
  cc = y[6][:Nn[6]]
  cc = cc[numpy.isfinite(cc)]

if rating > 7:
    d = y[rating-1][:Nn[7]]

#ABCDfile = scipy.io.loadmat("yabcd.mat")
#aaa1 = ABCDfile['aaa'][0]  
#aa1 = ABCDfile['aa'][0]
#bbb1 = ABCDfile['bbb'][0]
#bb1 = ABCDfile['bb'][0]
#b1 = ABCDfile['b'][0]
#cc1 = ABCDfile['cc'][0]
#a1 = ABCDfile['a'][0]
#d1 = ABCDfile['d'][0]

#for i in range(len(aaa)):
#    if (aaa[i] != aaa1[i]):
#        sys.stdout.write("aaa: %f aaa1: %f \n"%(aaa[i], aaa1[i]))
#
#for i in range(len(aa)):
#    if (aa[i] != aa1[i]):
#        sys.stdout.write("aa: %f aa1: %f \n"%(aa[i], aa1[i]))
#
#for i in range(len(bbb)):
#    if (bbb[i] != bbb1[i]):
#        sys.stdout.write("bbb: %f bbb1: %f \n"%(bbb[i], bbb1[i]))
#
#for i in range(len(bb)):
#    if (bb[i] != bb1[i]):
#        sys.stdout.write("bb: %f bb1: %f \n"%(bb[i], bb1[i]))
#
#for i in range(len(b)):
#    if (b[i] != b1[i]):
#        sys.stdout.write("b: %f b1: %f \n"%(b[i], b1[i]))
#
#for i in range(len(a)):
#    if (a[i] != a1[i]):
#        sys.stdout.write("a: %f a1: %f \n"%(a[i], a1[i]))
#
#if (d != d1):
#    sys.stdout.write("d: %f d1: %f \n"%(d, d1))
#
#for i in range(len(cc)):
#    if (cc[i] != cc1[i]):
#        sys.stdout.write("cc: %f cc1: %f \n"%(cc[i], cc1[i]))

minh = min(aaa)
maxh = max(aaa)
nbins = int ((maxh - minh) / step)
AAA, xh = numpy.histogram(aaa, bins=nbins, range=(minh, maxh))
Paaa = AAA/float(sum(AAA))

minh = min(aa)
maxh = max(aa)
nbins = int ((maxh - minh) / step)
AA, xh = numpy.histogram(aa, bins=nbins, range=(minh, maxh));
Paa = AA/float(sum(AA))

minh = min(a)
maxh = max(a)
nbins = int ((maxh - minh) / step)
A, xh = numpy.histogram(a, bins=nbins, range=(minh, maxh));
Pa = A/float(sum(A))

minh = min(bbb)
maxh = max(bbb)
nbins = int ((maxh - minh) / step)
BBB, xh = numpy.histogram(bbb, bins=nbins, range=(minh, maxh));
Pbbb = BBB/float(sum(BBB))

minh = min(bb)
maxh = max(bb)
nbins = int ((maxh - minh) / step)
BB, xh = numpy.histogram(bb, bins=nbins, range=(minh, maxh));
Pbb = BB/float(sum(BB))

minh = min(b)
maxh = max(b)
nbins = int ((maxh - minh) / step)
B, xh = numpy.histogram(b, bins=nbins, range=(minh, maxh));
Pb = B/float(sum(B))

minh = min(cc)
maxh = max(cc)
nbins = int ((maxh - minh) / step)
CC, xh = numpy.histogram(cc, bins=nbins, range=(minh, maxh));
Pcc = CC/float(sum(CC))

#ALLHfile = scipy.io.loadmat("allhist.mat")
#Paaa = ALLHfile['Paaa'][0]
#Paa = ALLHfile['Paa'][0]
#Pa = ALLHfile['Pa'][0]  
#Pbbb = ALLHfile['Pbbb'][0]  
#Pbb = ALLHfile['Pbb'][0]  
#Pb = ALLHfile['Pb'][0]  
#Pcc = ALLHfile['Pcc'][0]  

Taaa = sum((Paaa*(math.log(len(Paaa)))*Paaa))
Taa = sum((Paa*(math.log(len(Paa)))*Paa))
Ta = sum((Pa*(math.log(len(Pa)))*Pa))
Tbbb = sum((Pbbb*(math.log(len(Pbbb)))*Pbbb))
Tbb = sum((Pbb*(math.log(len(Pbb)))*Pbb))
Tb = sum((Pb*(math.log(len(Pb)))*Pb))
Tcc = sum((Pcc*(math.log(len(Pcc)))*Pcc))

Mean = [numpy.mean(aaa),numpy.mean(aa), \
        numpy.mean(a),numpy.mean(bbb),numpy.mean(bb), \
        numpy.mean(b),numpy.mean(cc)]

fval, pval = scipy.stats.f_oneway (aaa, aa, a, bbb, bb, b, cc)

#print "F-value: ", fval
#print "P value: ", pval

Ti = [Taaa, Taa, Ta, Tbbb, Tbb, Tb, Tcc]

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

run = 1000

X = numpy.random.rand(countries,tprev,run)
cdf = numpy.zeros((rating,rating), dtype='float64')

#Xfile = scipy.io.loadmat("XcdfPr.mat")
#X = Xfile['X']
#cdf = Xfile['cdf']
#Pr = Xfile['Pr']
#Mean = Xfile['Mean'][0]
#Ti = Xfile['Ti'][0]

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

#for j in range(run):
#   for i in range(countries):
#       for k in range(tprev):
#           sys.stdout.write ("%f "%X[i][k][j])
#       sys.stdout.write ("\n")
#   sys.stdout.write ("\n\n")

#exit(1);

for j in range(run):

    # da controllare
    for i in range(countries):
        x[i][0][j] = ms[i][time-1]

    for i in range (rating):
        cdf[i][0] = Pr[i][0]

    for i in range(rating):
        for k in range(1,rating):
            cdf[i][k] = Pr[i][k] + cdf[i][k-1]

    #for i in range(rating):
    #    for k in range(rating):
    #        sys.stdout.write ("%f "%cdf[i][k])
    #    sys.stdout.write ("\n")

    #for i in range(rating):
    #    for k in range(rating):
    #        sys.stdout.write ("%f "%Pr[i][k])
    #    sys.stdout.write ("\n")

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

    #for c in range(countries):
    #    for t in range(tprev):
    #        sys.stdout.write ("%d "%x[c][t][j])
    #    sys.stdout.write ("\n")

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

    #for t in range(tprev):
    #    sys.stdout.write ("%f "%r_prev[t][j])
    #    sys.stdout.write ("\n")

    #for i in range(rating):
    #   for t in range(tprev):
    #      sys.stdout.write ("%f "%tot[i][t][j])
    #   sys.stdout.write ("\n")

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

    #for t in range(tprev):
    #   sys.stdout.write ("%f "%entr[t][j])
    #sys.stdout.write ("\n")

    #for t in range(tprev):
    #    sys.stdout.write ("%f "%R_prev[t])
    #    sys.stdout.write ("\n")
   
    #exit(1)

    #print j, " of ", run
    sys.stdout.flush()

for t in range(tprev):
    entropia[t] =numpy.mean(entr[t])
    Var[t] = numpy.std(entr[t])

for t in range(tprev):
    print t+1, " ", entropia[t], " ", Var[t]

