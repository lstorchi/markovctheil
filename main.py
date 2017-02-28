import scipy.io
import numpy

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

print Pr
