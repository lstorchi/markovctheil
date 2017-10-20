import numpy.linalg
import numpy.random
import scipy.stats
import scipy.io
import argparse
import random
import numpy
import math
import sys
import os

import os.path

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import basicutils

countries = rm.shape[0]
   rating = numpy.max(rm)
   time = rm.shape[1]

 benchmark = numpy.amin(ir, 0)
 r = numpy.zeros((countries,time), dtype='float64')

 for i in range(countries):
       for j in range(time):
           r[i, j] = ir[i, j] - benchmark[j]

 R = numpy.sum(r, axis=0)
 Ri=numpy.sum(ir, axis=0)

 

 for t in range(time)
	for i in range(rating)
		for j in range(countries)
			if rm[j,t]==i
			    rc[i,t]=rc[i,t] + r[j,t] #totale credt spread pagati dalla classe di rating i
			    ri[i,t]= ri[i,t] + ir[i,t] #totale interest rates  pagati dalla classe di rating i
	
        s_r[i,t]=rc[i,t] / R[t]
	s_i[i,t]=ri[i,t] / Ri[t]
 
 s=0
 for s in range(0,10)
	for t in range(time)
            for i in range(rating)
		p_s[i,t,s]=s_i[i,t]**s
	r_s=numpy.sum(p_s, axis=0)

      	    E_r[i,t,s]= p_s[i,t,s]/r_s[t,s]

	for t in range(time)
            for i in range(rating)
		if E_r !=0
		   T[t,s]+=E_r[i,t,s]*math.log(float(countries) * E_r[i,t,s]

  



