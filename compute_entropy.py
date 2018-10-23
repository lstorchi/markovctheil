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

namems = "mtx"

r = msd[namems] #traiettorie spread simulate

countries = r.shape[0]
time = r.shape[1]
run = r.shape[2]
print countries, time, run

R = numpy.sum(r, axis=0)
Rm = numpy.zeros((time), dtype = 'float64')
pm = numpy.zeros((countries,time), dtype = 'float64')
entropy = numpy.zeros((time), dtype = 'float64')
std = numpy.zeros((time), dtype = 'float64')
T = numpy.zeros((time, run), dtype = 'float64')

for i in range(run):
    for j in range(countries):
        for t in range(time):
            p = r[j,t,i] / R[t,i]
            #pm[j,t] = numpy.mean(p[j,t])
            if p != 0:
                T[t,i] += p*math.log(float(countries) * p)


for t in range(time):
    entropy[t] = numpy.mean(T[t])
    std[t] = numpy.std(T[t])

basicutils.mat_to_file(entropy, 'edt.txt')
basicutils.mat_to_file(std, 'std.txt')

plt.plot(entropy)
plt.show()


