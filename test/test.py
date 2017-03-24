import numpy.linalg
import numpy.random
import scipy.stats
import scipy.io
import numpy
import math
import sys
import os

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

rating = 2
Pr = numpy.zeros((rating,rating), dtype='float64')

alpha = 0.8
beta = 0.3

Pr[0][0] = 1.0 - alpha
Pr[0][1] = alpha

Pr[1][0] = beta
Pr[1][1] = 1.0 - beta

ai = numpy.identity(rating, dtype='float64') - numpy.matrix.transpose(Pr)    
a = numpy.zeros((rating+1,rating), dtype='float64')
for i in range(rating):
    for j in range(rating):
        a[i][j] = ai[i][j]
for i in range(rating):
    a[rating][i] = 1.0 


print a

#exit()

b = numpy.zeros(rating+1, dtype='float64')
b[rating] = 1.0
#x = numpy.linalg.solve(a, b)
x = numpy.linalg.lstsq(a, b)
print x[0]
#print x[0][0] * Pr

#print numpy.linalg.matrix_power(Pr, 20000)
