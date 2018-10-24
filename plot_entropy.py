import scipy
import numpy
import math
import sys 
import scipy.stats 
import scipy.io 

import matplotlib.pyplot as plt
sys.path.append("./module")
import basicutils 

if len(sys.argv) != 3:
    print "need to specify a filename and a run"
    exit(1)

filename = sys.argv[1]
ri = int(sys.argv[2])

msd = scipy.io.loadmat(filename)

namems = "mtx"

rin = msd[namems] #traiettorie spread simulate

countries = rin.shape[0]
time = rin.shape[1]
run = rin.shape[2]
print "countries: ", countries, "time: ", time, "run: ", run

r = numpy.zeros((countries, time), dtype = 'float64')
r = rin[:, :, ri] 

print r

entropy = numpy.zeros((time), dtype = 'float64')
std = numpy.zeros((time), dtype = 'float64')
T = numpy.zeros((time), dtype = 'float64')

p = numpy.zeros((countries, time), dtype = 'float64')

R = numpy.sum(r, axis=0)

for c in range(countries):
    print c+1 , " of ", countries
    for t in range(time):
        p[c, t] = r[c,t] / R[t]

for t in range(time):
    print t+1 , " of ", time
    
    for c in range(countries):
        T[t] += p[c, t] * math.log(float(countries) * p[c, t])


plt.plot(T)
plt.show()
