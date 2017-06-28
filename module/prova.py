
import numpy.linalg
import numpy.random
import numpy.stats
import scipy.io
import argparse
import numpy
import math
import sys 
import os

rm rating credit rating.xlsx


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
v=sum(change[v0])

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



ir bp.xls 
   
   npr = pr - numpy.identity(rating, dtype='float64')
   s, v, d = numpy.linalg.svd(npr)
   
   if verbose:
       print (" ")
       print ("mean value: ", numpy.mean(v))
   
   for i in range(len(ir)):
       for j in range(len(ir[0])):
           if math.isnan(ir[i, j]):
              ir[i, j] = float('inf')
   
   benchmark = numpy.amin(ir, 0)
   
   r = numpy.zeros((countries,time), dtype='float64') 
   
   for i in range(countries):
       for j in range(time):
           r[i, j] = ir[i, j] - benchmark[j]
   
   for i in range(len(r)):
       for j in range(len(r[0])):
           if (r[i, j] == float('Inf')):
              r[i, j] = float('nan')
   
   ist = numpy.zeros((rating,time*countries), dtype='float64')
   nn = numpy.zeros((rating), dtype='int')

   if setval != None:
     setval.setValue(50.0)
     if setval.wasCanceled():
         errmsg.append("Cancelled!")
         return False
   
   for i in range(rating):
       for j in range(countries):
           for k in range(time):
               if rm[j, k] == i+1: 
                   nn[i] = nn[i] + 1 
                   ist[i, nn[i]-1] = r[j, k]
   
   y = numpy.zeros((ist.shape[0], nn[0]), dtype='float64')
   for i in range(len(ist)):
       y[i] = ist[i, 0:nn[0]]
   
   tiv = []

   fname = ""
   
   if rating > 0:
       if outfiles:
           fname = "aaa"
 
       a, b, c, d, e = basicutils.extract_ti_mean (y[0, :nn[0]], step, 0, numofrun, \
               fname)

       allratingsbins.append(d)
       allratings.append(a)
       stdeval.append(e)
       meanval.append(b)
       tiv.append(c)
   
   if rating > 1:
       if outfiles:
           fname = "aa"

       a, b, c, d, e = basicutils.extract_ti_mean (y[1, :nn[1]], step, 1, numofrun, \
               fname)

       allratingsbins.append(d)
       allratings.append(a)
       stdeval.append(e)
       meanval.append(b)
       tiv.append(c)
   
   if rating > 2:
       if outfiles:
           fname = "a"
 
       a, b, c, d, e = basicutils.extract_ti_mean (y[2, :nn[2]], step, 2, numofrun, \
               fname)

       allratingsbins.append(d)
       allratings.append(a)
       stdeval.append(e)
       meanval.append(b)
       tiv.append(c)
   
   if rating > 3: 
       if outfiles:
           fname = "bbb"
 
       a, b, c, d, e = basicutils.extract_ti_mean (y[3, :nn[3]], step, 3, numofrun, \
               fname)

       allratingsbins.append(d)
       allratings.append(a)
       stdeval.append(e)
       meanval.append(b)
       tiv.append(c)
   
   if rating > 4:
       if outfiles:
           fname = "bb"
 
       a, b, c, d, e = basicutils.extract_ti_mean (y[4, :nn[4]], step, 4, numofrun, \
               fname)

       allratingsbins.append(d)
       allratings.append(a)
       stdeval.append(e)
       meanval.append(b)
       tiv.append(c)
   
   if rating > 5:
       if outfiles:
           fname = "b"
 
       a, b, c, d, e = basicutils.extract_ti_mean (y[5, :nn[5]], step, 5, numofrun, \
               fname)

       allratingsbins.append(d)
       allratings.append(a)
       stdeval.append(e)
       meanval.append(b)
       tiv.append(c)
   
   if rating > 6:
       if outfiles:
           fname = "cc"
 
       a, b, c, d, e = basicutils.extract_ti_mean (y[6, :nn[6]], step, 6, numofrun, \
               fname)

       allratingsbins.append(d)
       allratings.append(a)
       stdeval.append(e)
       meanval.append(b)
       tiv.append(c)
   
   if rating > 7:
       if outfiles:
           fname = "d"
 
       a, b, c, d, e = basicutils.extract_ti_mean (y[7, :nn[7]], step, 7, numofrun, \
               fname)

       allratingsbins.append(d)
       allratings.append(a)
       stdeval.append(e)
       meanval.append(b)
       tiv.append(c)
   
   fval = 0.0
   pval = 0.0

   if setval != None:
     setval.setValue(75.0)
     if setval.wasCanceled():
         errmsg.append("Cancelled!")
         return False
   
   args = [] 

   for i in range(len(allratings)):
       args.append(allratings[i])

   fval, pval = scipy.stats.f_oneway (*args)

   if verbose:
     print (" ")
   
   oufilename = "1wayanova_"+str(numofrun)+".txt"
   
   if outfiles:
     if os.path.exists(oufilename):
         os.remove(oufilename)
   
     outf = open(oufilename, "w")
   
     outf.write("F-value: %f\n"%fval)
     outf.write("P value: %f\n"%pval)
   
     outf.close()
   
   s_t = numpy.zeros((countries,time), dtype='float64')
   
   for i in range(r.shape[0]):
       for j in range(r.shape[1]):
           if math.isnan(r[i, j]):
               r[i, j] = 0.0
   
   R_t = numpy.sum(r, axis=0)
   T_t = numpy.zeros(time, dtype='float64')
   
   for t in range(time):
       for k in range(countries):
           s_t[k, t] = r[k, t] / R_t[t]
           if s_t[k, t] != 0:
               T_t[t] += s_t[k, t]*math.log(float(countries) * s_t[k, t])
   
   oufilename = "entropy_histi_"+str(numofrun)+".txt"
   
   if outfiles:
     basicutils.vct_to_file(T_t, oufilename)

   if setval != None:
     setval.setValue(100.0)
     if setval.wasCanceled():
         errmsg.append("Cancelled!")
         return False
   
