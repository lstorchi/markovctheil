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

# 0.5 + ((2.82 - 0.0) /0.25) ==> int
AAA = numpy.histogram(aaa,11);

print AAA

#Paaa = AAA/sum(AAA);
#AA = hist(aa,0:0.25:5.69);
#Paa = AA/sum(AA);
#A = hist(a,0:0.25:11.57);
#Pa = A/sum(A);
#BBB = hist(bbb,0.13:0.25:11.29);
#Pbbb = BBB/sum(BBB);
#BB = hist(bb,1.52:0.25:12.15);
#Pbb = BB/sum(BB);
#B = hist(b,9.23:0.25:12.93);
#Pb = B/sum(B);
#CC = hist(cc,4.67:0.25:27.40);
#Pcc = CC/sum(CC);

#

