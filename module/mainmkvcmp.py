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

#####################################################################

def evolve_country (mc, c, tstart, endtime, cdf, ratidx, rating, \
        tprev):

   rnd = 0.0
   while rnd == 0.0:
       rnd = random.random()
   
   for j in range(rating):
       if rnd <= cdf[ratidx-1, j]:
           for t in range(tstart+1, endtime):
               mc[c, t] = mc[c, tstart]
           
           if endtime < tprev:
               mc[c, endtime] = j + 1
           
           break

#####################################################################

def main_mkc_comp (rm, ir, timeinf, step, tprev, \
        numofrun, verbose, outfiles, seed, errmsg, \
        entropia, var, allratings, allratingsbins, \
        pr, meanval, stdeval, \
        setval=None):

   if seed:
       numpy.random.seed(9001)

   countries = rm.shape[0]
   rating = numpy.max(rm)
   time = rm.shape[1]
   
   if (rating <= 0) or (rating > 8):
       errmsg.append("rating " + rating + " is not a valid value")
       return False
   
   nk = numpy.zeros((rating,rating,countries), dtype='int64')
   num = numpy.zeros((rating,rating), dtype='int64')
   den = numpy.zeros(rating, dtype='int64')

   if setval != None:
        setval.setValue(0)
        setval.setLabelText("Historical data analysis")
   
   for k in range(countries):
       for t in range(time-1):
           for i in range(rating):
               for j in range(rating):
                   if (rm[k, t] == (i+1)) and (rm[k, t+1] == (j+1)):
                       nk[i, j, k] = nk[i, j, k] + 1
   
                   num[i, j] = sum(nk[i, j])
   
               den[i] = sum(num[i])
   
       if verbose:
         basicutils.progress_bar(k+1, countries)

       if setval != None:
         setval.setValue(100.0*(float(k+1)/float(countries)))
         if setval.wasCanceled():
             errmsg.append("Cancelled!")
             return False

   if setval != None:
        setval.setValue(0)
        setval.setLabelText("Running...")
   
   for i in range(rating):
       for j in range(rating):
           if den[i] != 0:
               pr[i, j] = float(num[i, j])/float(den[i])
           else: 
               pr[i, j] = 0.0
   
   if timeinf: # matrice delle probabilita' diventa stazionaria tempo elevato 
       if verbose:
         print ("")
         print ("Solve ...")

       ai = numpy.identity(rating, dtype='float64') - numpy.matrix.transpose(pr)
       a = numpy.zeros((rating+1,rating), dtype='float64')
   
       for i in range(rating):
           for j in range(rating):
               a[i, j] = ai[i, j]
   
       for i in range(rating):
           a[rating, i] = 1.0 
   
       b = numpy.zeros(rating+1, dtype='float64')
       b[rating] = 1.0
       x = numpy.linalg.lstsq(a, b)
       
       for j in range(rating):
           for i in range(rating):
               pr[i, j] = x[0][j] 

   if verbose:
     print (" ")
     print ("Solve SVD ")
   
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
   
   bp = numpy.zeros((countries,tprev,numofrun), dtype='float64')
   tot = numpy.zeros((rating,tprev,numofrun), dtype='float64')
   ac = numpy.zeros((rating,tprev,numofrun), dtype='float64')
   xm = numpy.zeros((countries,tprev), dtype='float64')
   cdf = numpy.zeros((rating,rating), dtype='float64')
   cont = numpy.zeros((rating,tprev,numofrun), dtype='int')
   r_prev = numpy.zeros((tprev,numofrun), dtype='float64')
   term = numpy.zeros((tprev,numofrun), dtype='float64')
   entr = numpy.zeros((tprev,numofrun), dtype='float64')
   t1 = numpy.zeros((tprev,numofrun), dtype='float64')
   t2 = numpy.zeros((tprev,numofrun), dtype='float64')

 
   for i in range (rating):
       cdf[i, 0] = pr[i, 0]
   
   for i in range(rating):
       for j in range(1,rating):
           cdf[i, j] = pr[i, j] + cdf[i, j-1]

   if setval != None:
        setval.setValue(0)
        setval.setLabelText("Monte Carlo simulation")
   
   for run in range(numofrun):

       x = numpy.zeros((countries,tprev), dtype='int')
       xi = numpy.random.rand(countries,tprev)
   
       for c in range(countries):
           x[c, 0] = rm[c, time-1]
   
       for c in range(countries):
           if xi[c, 0] <= cdf[x[c, 0]-1, 0]:
               x[c, 1] = 1
   
           for k in range(1,rating):
               if (cdf[x[c, 0]-1, k-1] < xi[c, 0]) and \
                       (xi[c, 0] <= cdf[x[c, 0]-1, k] ):
                  x[c, 1] = k + 1
   
           for t in range(2,tprev):
               if xi[c, t-1] <= cdf[x[c, t-1]-1, 0]:
                   x[c, t] = 1
   
               for k in range(1,rating):
                   if (cdf[x[c, t-1]-1, k-1] < xi[c, t-1]) \
                           and (xi[c, t-1] <= cdf[x[c, t-1]-1, k]):
                     x[c, t] = k + 1
   
       for t in range(tprev):
           for c in range(countries):
               for i in range(rating):
                   if x[c, t] == i+1:
                       bp[c, t, run] = meanval[i]
                       cont[i, t, run] = cont[i, t, run] + 1
                       tot[i, t, run] = cont[i, t, run] * meanval[i]
               
           summa = 0.0
           for a in range(bp.shape[0]):
               summa += bp[a, t, run]
           r_prev[t, run] = summa
   
       for t in range(tprev):
           for i in range(rating):
                ac[i, t, run] = tot[i, t, run]/r_prev[t, run]
                if ac[i, t, run] != 0.0:
                    t1[t, run] += (ac[i, t, run]*tiv[i])
                    t2[t, run] += (ac[i, t, run]*math.log(float(rating)*ac[i, t, run]))
                    if cont[i, t, run] != 0:
                       term[t, run] += ac[i, t, run]* \
                               math.log(float(countries)/(float(rating)*cont[i, t, run]))
    
           entr[t, run] = t1[t, run] + t2[t, run] + term[t, run]
   
       if verbose:
           basicutils.progress_bar(run+1, numofrun)

       if setval != None:
           setval.setValue(100.0*(float(run+1)/float(numofrun)))
           if setval.wasCanceled():
             errmsg.append("Cancelled!")
             return False
   
   if verbose:
     print (" ")
   
   oufilename = "entropy_"+str(numofrun)+".txt"

   for t in range(tprev):
       entropia[t] =numpy.mean(entr[t])
       var[t] = numpy.std(entr[t])

   if outfiles:
   
     if os.path.exists(oufilename):
         os.remove(oufilename)
   
     outf = open(oufilename, "w")
  
     for t in range(tprev):
         outf.write("%d %f %f \n"%(t+1, entropia[t], var[t]))
    
     outf.close()
   
   acm = numpy.zeros((rating,tprev), dtype='float64')
   for i in range(acm.shape[0]):
       for j in range(acm.shape[1]):
           acm[i, j] = numpy.mean(ac[i, j])
   
   oufilename = "acm_"+str(numofrun)+".txt"
   
   if outfiles:
     basicutils.mat_to_file (acm, oufilename)
   
   bpm = numpy.zeros((countries,tprev), dtype='float64')
   for i in range(bpm.shape[0]):
       for j in range(bpm.shape[1]):
           bpm[i, j] = numpy.mean(bp[i, j])
   
   oufilename = "bpm_"+str(numofrun)+".txt"
  
   if outfiles:
     basicutils.mat_to_file (bpm, oufilename)

   return True

#####################################################################

def main_mkc_comp_cont (rm, ir, timeinf, step, tprev, \
        numofrun, verbose, outfiles, seed, errmsg, \
        entropia, var, allratings, allratingsbins, \
        pr, meanval, stdeval, \
        setval=None):

   if seed:
       numpy.random.seed(9001)

   countries = rm.shape[0]
   time = rm.shape[1]
   rating = numpy.max(rm)
  
   #print "time: ", time
   #print "rating: ", rating
   #print "countries: ", countries

   nk = numpy.zeros((rating, rating, countries), dtype='float64')
   num = numpy.zeros((rating, rating), dtype='float64')
   change = numpy.zeros((countries, rating), dtype='float64')
   amtx = numpy.zeros((rating, rating), dtype='float64')

   #print rm
   
   for c in range(countries):
       v0 = rm[c,0]
       ts = 0.0e0
       for t in range(time):
           if (rm[c,t] != v0):
               change[c,v0-1] += ts
               v0 = rm[c,t]
               ts = 0.0e0
           else:
               ts = ts + 1.0e0

       change[c,v0-1] = change[c, v0-1] + ts;

   #print change

   v = numpy.sum(change, axis=0)

   for c in range(countries):
       for t in range(time-1):
           for i in range (rating):
               for j in range(rating):
                   if (rm[c,t] == i+1) and (rm[c,t+1] == j+1):
                       nk[i,j,c] = nk[i,j,c] + 1.0e0

       if verbose:
         basicutils.progress_bar(c+1, countries)
 
                   
   for i in range(nk.shape[0]):
       for j in range(nk.shape[1]):
           val = 0.0e0
           for c in range(nk.shape[2]):
               val += nk[i,j,c]
           num[i,j] = val
   
   #print 'num of transition'
   #print num
   #print "v: ", v

   for i in range(rating):
       for j in range(rating):
           if i != j:
               amtx[i,j] = num[i,j]/v[i]
           
   q = numpy.sum(amtx, axis=1)
   for i in range(rating):
       amtx[i, i] = -1.0e0 * q[i] 

   testrow = numpy.sum(amtx, axis=1)
   for t in testrow:
       if math.fabs(t) > 1e-19 :
           print "Error in A matrix "
           exit(1)

   #print testrow

   #print "A: "
   #print amtx

   if outfiles:
       basicutils.mat_to_file (amtx, "amtx_"+str(numofrun)+".txt")
   
   for t in range(time):
       pr[:,:,t] = scipy.linalg.expm(t*amtx)

   for t in range(pr.shape[2]):
       testrow = numpy.sum(pr[:,:,t], axis=1)
       for v in testrow:
           diff = math.fabs(v - 1.0) 
           if diff > 5e-13 :
               print "Error in PR matrix at ", t+1, " diff ", diff
               print testrow
               exit(1)
   
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

   if outfiles:
       oufname = "mean_stdval_"+str(numofrun)+".txt"
       fp = open(oufname, "w")
       for i in range(len(meanval)):
           fp.write("%2d %15.7f %15.7f \n"%(i+1, meanval[i], stdeval[i]))
       fp.close()
   
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

   pmtx = numpy.zeros((rating, rating), dtype='float64')

   for x in range(rating):
       for y in range(rating):
           if x != y:
               if q[x] > 0.0:
                   pmtx[x,y] = amtx[x,y] / q[x]
           else:
               pmtx[x,y] = 0.0

   if outfiles:
       basicutils.mat_to_file (pmtx, "pmtx_"+str(numofrun)+".txt")

   cdf = numpy.zeros((rating, rating), dtype='float64')
   mc =  numpy.zeros((countries,tprev), dtype='int')
   bp = numpy.zeros((countries,tprev,numofrun), dtype='float64')
   cont = numpy.zeros((rating,tprev,numofrun), dtype='int')
   r_prev = numpy.zeros((tprev,numofrun), dtype='float64')
   tot = numpy.zeros((rating,tprev,numofrun), dtype='float64')
   ac = numpy.zeros((rating,tprev,numofrun), dtype='float64')
   term = numpy.zeros((tprev,numofrun), dtype='float64')
   entr = numpy.zeros((tprev,numofrun), dtype='float64')
   t1 = numpy.zeros((tprev,numofrun), dtype='float64')
   t2 = numpy.zeros((tprev,numofrun), dtype='float64')

   for x in range(rating):
       cdf[x,0] = pmtx[x,0]
       for y in range(1,rating):
           cdf[x,y] = pmtx[x,y] + cdf[x,y-1]

   for x in range(rating):
       cdf[x,x] = 0.0

   counter = 0
   mcount = 0

   for run in range(numofrun):

       for i in range(rm.shape[0]):
           mc[i, 0] = rm[i, rm.shape[1]-1]
       
       endtimexc = numpy.zeros(countries, dtype='int')
       
       for c in range(countries):
           todo = True
           
           rnumb = 0.0
           while rnumb == 0.0:
               rnumb = random.random()
 
           while todo:
              tstart = endtimexc[c]
              startrating = mc[c, tstart]
              
              if q[startrating-1] != 0.0:
                  invx = -1.0 * math.log(1.0 - rnumb, \
                          math.e) / q[startrating-1]
              else:
                  invx = float(tprev + 1)
              
              iinvx = int(invx + 0.5)
              if iinvx == 0:
                  iinvx = 1

              if (iinvx + tstart) >= tprev:
                  for t in range(tstart, tprev):
                      mc[c, t] = mc[c, tstart]
                  todo = False
              else:
                  endtime = int(iinvx) + tstart
                  
                  evolve_country (mc, c, tstart, endtime, cdf, startrating, \
                          rating, tprev)
              
                  endtimexc[c] = endtime

                  if endtime >= tprev:
                      todo = False

       counter += 1
       if counter == 100:
           counter = 0
           if outfiles:
               mcount += 1
               basicutils.mat_to_file (mc, "mc_"+str(numofrun)+"_"+\
                       str(mcount)+".txt")

       """ 
       for c in range(countries):
           oldv = mc[c, 0]
           sys.stdout.write( "Cont: %d startr: %f ==> "%(c, oldv))
           for j in range(tprev):
               if mc[c, j] != oldv:
                 sys.stdout.write( "[%d] %f "%(j , mc[c, j]))
                 oldv = mc[c, j]
       
           print " lastv: ", mc[c, tprev-1]
       """
   
       for t in range(tprev):
           for c in range(countries):
               for i in range(rating):
                   if mc[c, t] == i+1:
                       bp[c, t, run] = meanval[i]
                       cont[i, t, run] = cont[i, t, run] + 1
                       tot[i, t, run] = cont[i, t, run] * meanval[i]
               
           summa = 0.0
           for a in range(bp.shape[0]):
               summa += bp[a, t, run]
           r_prev[t, run] = summa
   
       for t in range(tprev):
           for i in range(rating):
                ac[i, t, run] = tot[i, t, run]/r_prev[t, run]
                if ac[i, t, run] != 0.0:
                    t1[t, run] += (ac[i, t, run]*tiv[i])
                    t2[t, run] += (ac[i, t, run]*math.log(float(rating)*ac[i, t, run]))
                    if cont[i, t, run] != 0:
                       term[t, run] += ac[i, t, run]* \
                               math.log(float(countries)/(float(rating)*cont[i, t, run]))
    
           entr[t, run] = t1[t, run] + t2[t, run] + term[t, run]
   
       if verbose:
           basicutils.progress_bar(run+1, numofrun)

       if setval != None:
           setval.setValue(100.0*(float(run+1)/float(numofrun)))
           if setval.wasCanceled():
             errmsg.append("Cancelled!")
             return False
   
   if verbose:
     print (" ")
   
   oufilename = "entropy_"+str(numofrun)+".txt"

   for t in range(tprev):
       entropia[t] =numpy.mean(entr[t])
       var[t] = numpy.std(entr[t])

   if outfiles:
   
     if os.path.exists(oufilename):
         os.remove(oufilename)
   
     outf = open(oufilename, "w")
  
     for t in range(tprev):
         outf.write("%d %f %f \n"%(t+1, entropia[t], var[t]))
    
     outf.close()
   
   acm = numpy.zeros((rating,tprev), dtype='float64')
   for i in range(acm.shape[0]):
       for j in range(acm.shape[1]):
           acm[i, j] = numpy.mean(ac[i, j])
   
   oufilename = "acm_"+str(numofrun)+".txt"
   
   if outfiles:
     basicutils.mat_to_file (acm, oufilename)
   
   bpm = numpy.zeros((countries,tprev), dtype='float64')
   for i in range(bpm.shape[0]):
       for j in range(bpm.shape[1]):
           bpm[i, j] = numpy.mean(bp[i, j])
   
   oufilename = "bpm_"+str(numofrun)+".txt"
  
   if outfiles:
     basicutils.mat_to_file (bpm, oufilename)

   return True
