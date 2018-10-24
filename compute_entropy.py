import scipy
import numpy
import math
import sys 
import scipy.stats 
import scipy.io 

import matplotlib.pyplot as plt
sys.path.append("./module")
import basicutils 

if len(sys.argv) != 2:
    print "need to specify a filename"
    exit(1)

filename=sys.argv[1]

msd = scipy.io.loadmat(filename)

namems = "spread_synth"

r = msd[namems] #traiettorie spread simulate

run = r.shape[0]
time = r.shape[1]
countries = r.shape[2]
print "countries: ", countries, "time: ", time, "run: ", run

#exit(1)

R = numpy.sum(r, axis=2)

#print r[0,1,:]
#print r[1,1,:]
#print R[1]

print "R.shape: ", R.shape

entropy = numpy.zeros((time), dtype = 'float64')
std = numpy.zeros((time), dtype = 'float64')
T = numpy.zeros((time, run), dtype = 'float64')

p = numpy.zeros((countries, time, run), dtype = 'float64')

for c in range(countries):
    print c+1 , " of ", countries
    for t in range(time):
        for i in range(run):
            p[c, t, i] = r[i,t,c] / R[i, t]

for t in range(time):
    print t+1 , " of ", time
    for i in range(run):

        for c in range(countries):
            T[t,i] += p[c, t, i] * math.log(float(countries) * p[c, t, i])

for t in range(time):
    entropy[t] = numpy.mean(T[t, :])
    std[t] = numpy.std(T[t, :])

pm=numpy.zeros((countries, time), dtype = 'float64')
entropy_pm = numpy.zeros((time), dtype = 'float64')

for c in range(countries):
    for t in range(time):
        pm[c,t] = numpy.mean(p[c,t,:])

for t in range(time):
    for c in range(countries):
        entropy_pm[t] += pm[c, t] * math.log(float(countries) * pm[c, t])

basicutils.vct_to_file(entropy, 'edt.txt')
basicutils.vct_to_file(std, 'std.txt')
basicutils.vct_to_file(entropy_pm, 'meanp_entropy.txt')

plt.plot(entropy)
plt.plot(entropy_pm)
#plt.plot(std)
plt.show()

for c in range(countries):
    plt.plot(pm[c,:])
plt.show()
