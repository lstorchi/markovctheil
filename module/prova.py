
import numpy.linalg
import numpy.random
import numpy.stats
import scipy.io
import argparse
import numpy
import math
import sys 
import os

countries=rm.shape[0]
time=len(rm[1,:])
rating=numpy.max(rm)

nk=numpy.zeros((rating,rating,countrie), dtype='int64')
num=numpy.zerps((rating,rating), dtype='int64')
change=numpy.zeros((countries,rating), dtype='int64')
a=numpy.zeros((rating,rating), dtype='int64')
pr=numpy.zeros((rating,rating), dtype='int64')

for c in range(countries)
    v0=data[c,0]
    ts=0
    for t in range(time)
       if (data[c,t]!=v0)
          change[c,v]=change[c,v0]+ts
       else
	  ts=ts+1
v=numpy.sum(change[v0])

for c in range(countries) 
    for t in range(time-1)
        for i in range (rating)
	    for j in range(rating)
                if (data[c,t]==i+1) and (data[c,t+1]=j+1)
		    nk[i,j,c]=nk[i,j,c]+1
	        num[i,j]= sum(nk[i,j])

print 'num of transition', num

for i in range(rating)
    for j in range(rating)
	if i!=j
	   a(i,j)=float(num[i,j]/v[i])
for i in range(rating)
    a[i,i]= float(-sum(a[i])
for t in range(time)
    pr[i,j,t]= scipy.linalg.expm(t*a)

print 'pr', pr

