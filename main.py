import numpy.linalg
import scipy.io
import numpy
import math

msd = scipy.io.loadmat("ms.mat")
bpd = scipy.io.loadmat("bp.mat")

#print msd['ms']
#print bpd['i_r']

ms = msd['ms']
i_r = bpd['i_r']

time = len(ms[1,:])

Nk = numpy.zeros((8,8,26), dtype='int64')
Num = numpy.zeros((8,8), dtype='int64')
Den = numpy.zeros(8, dtype='int64')
Pr = numpy.zeros((8,8), dtype='float64')

for k in range(26):
    for time in range(226):
        for i in range(8):
            for j in range(8):
                if (ms[k][time] == (i+1)) and (ms[k][time+1] == (j+1)):
                    Nk[i][j][k] = Nk[i][j][k] + 1

                Num[i][j] = sum(Nk[i][j])

            Den[i] = sum(Num[i])

    print k , " of 26"

#print Num 
#print Den

for i in range(8):
    for j in range(8):
        if Den[i] != 0:
            Pr[i][j] = float(Num[i][j])/float(Den[i])
        else: 
            Pr[i][j] = 0.0

newPr = Pr - numpy.identity(8, dtype='float64')

#print newPr

s, v, d = numpy.linalg.svd(newPr)
#print numpy.mean(v)

for i in range(len(i_r)):
    for j in range(len(i_r[0])):
        if math.isnan(i_r[i][j]):
           i_r[i][j] = float('inf')

benchmark = numpy.amin(i_r, 0)

r = numpy.zeros((26,227), dtype='float64') 

for k in range(26):
    for Time in range(227):
        r[k][Time] = i_r[k][Time] - benchmark[Time]

for i in range(len(r)):
    for j in range(len(r[0])):
        if (r[i][j] == float('Inf')):
           r[i][j] = float('nan')

#print r

ist = numpy.zeros((8,227*26), dtype='float64')
Nn = numpy.zeros((8,1), dtype='int')

for i in range(8):
    for k in range(26):
        for Time in range(227):
            if ms[k][Time] == i+1: 
                Nn[i] = Nn[i] + 1 
                ist[i][Nn[i]-1] = r[k][Time]

#print Nn
#print ist.shape

Mean = numpy.zeros((7), dtype='float64')
y = numpy.zeros((ist.shape[0], 2135), dtype='float64')
for i in range(len(ist)):
    y[i] = ist[i][0:2135]

aaa = y[0]
aaa = aaa[numpy.isfinite(aaa)]

aa = y[1][:879]
aa = aa[numpy.isfinite(aa)]

a = y[2][:1248]
a = a[numpy.isfinite(a)]

bbb = y[3][:1033]
bbb = bbb[numpy.isfinite(bbb)]

bb = y[4][:410]
bb = bb[numpy.isfinite(bb)]

b = y[5][:131]
b = b[numpy.isfinite(b)]

cc = y[6][:66]
cc = cc[numpy.isfinite(cc)]

d = y[7][0]

minh = (0.0-(0.25/2.0))
maxh = (2.82+(0.25/2.0))
nbins = int ((maxh - minh) / 0.25)
AAA, xh = numpy.histogram(aaa, bins=nbins, range=(minh, maxh))
Paaa = AAA/float(sum(AAA))

minh = (0.0-(0.25/2.0))
maxh = (5.69+(0.25/2.0))
nbins = int ((maxh - minh) / 0.25)
AA, xh = numpy.histogram(aa, bins=nbins, range=(minh, maxh));
Paa = AA/float(sum(AA))

minh = (0.0-(0.25/2.0))
maxh = (11.57+(0.25/2.0))
nbins = int ((maxh - minh) / 0.25)
A, xh = numpy.histogram(a, bins=nbins, range=(minh, maxh));
Pa = A/float(sum(A))

minh = (0.13-(0.25/2.0))
maxh = (11.29+(0.25/2.0))
nbins = int ((maxh - minh) / 0.25)
BBB, xh = numpy.histogram(bbb, bins=nbins, range=(minh, maxh));
Pbbb = BBB/float(sum(BBB))

minh = (1.52-(0.25/2.0))
maxh = (12.15+(0.25/2.0))
nbins = int ((maxh - minh) / 0.25)
BB, xh = numpy.histogram(bb, bins=nbins, range=(minh, maxh));
Pbb = BB/float(sum(BB))

minh = (9.23-(0.25/2.0))
maxh = (12.93+(0.25/2.0))
nbins = int ((maxh - minh) / 0.25)
B, xh = numpy.histogram(b, bins=nbins, range=(minh, maxh));
Pb = B/float(sum(B))

minh = (4.67-(0.25/2.0))
maxh = (27.40+(0.25/2.0))
nbins = int ((maxh - minh) / 0.25)
CC, xh = numpy.histogram(cc, bins=nbins, range=(minh, maxh));
Pcc = CC/float(sum(CC))
