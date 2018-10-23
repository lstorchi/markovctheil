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
print "countries: ", countries, "time: ", time, "run: ", run

R = numpy.sum(r, axis=0)
print "R.shape: ", R.shape

Rm = numpy.zeros((time), dtype = 'float64')
entropy = numpy.zeros((time), dtype = 'float64')
std = numpy.zeros((time), dtype = 'float64')
T = numpy.zeros((time, run), dtype = 'float64')


meanp = numpy.zeros((countries,time), dtype = 'float64')


for j in range(countries):

    for t in range(time):
        pm = numpy.float64(0.0)

        for i in range(run):
            p = r[j,t,i] / R[t,i]

            pm += p

            if p != 0.0:
                T[t,i] += p*math.log(float(countries) * p)
        
        pm = pm / numpy.float64(run) 

        meanp[j, t] = pm

for t in range(time):
    entropy[t] = numpy.mean(T[t, :])
    std[t] = numpy.std(T[t, :])

basicutils.vct_to_file(entropy, 'edt.txt')
basicutils.vct_to_file(std, 'std.txt')
basicutils.mat_to_file(meanp, 'meanp.txt')

plt.plot(entropy)
plt.show()


