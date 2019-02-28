import numpy.linalg
import numpy.random
import scipy.stats
import scipy.io
import argparse
import numpy
import math
import sys
import os

import os.path

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

import basicutils

#####################################################################

def main_mkc_comp (rm, ir, timeinf, step, tprev, \
        numofrun, verbose, outfiles, seed, errmsg, \
        entropia, var, allratings, allratingsbins, \
        pr, meanval, stdeval, usecopula, setval=None):

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
         print ""
         print "Solve ..."

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
     print " "
     print "Solve SVD "
   
   npr = pr - numpy.identity(rating, dtype='float64')
   s, v, d = numpy.linalg.svd(npr)
   
   if verbose:
       print " "
       print "mean value: ", numpy.mean(v)
   
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
     print " "
   
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
   ac = numpy.zeros((rating,tprev,numofrun), dtype='float64')
   xm = numpy.zeros((countries,tprev), dtype='float64')
   cdf = numpy.zeros((rating,rating), dtype='float64')
   x = numpy.zeros((countries,tprev), dtype='int')
   r_prev = numpy.zeros((tprev,numofrun), dtype='float64')
   term = numpy.zeros((tprev,numofrun), dtype='float64')
   entr = numpy.zeros((tprev,numofrun), dtype='float64')
   t1 = numpy.zeros((tprev,numofrun), dtype='float64')
   t2 = numpy.zeros((tprev,numofrun), dtype='float64')

   G = None 
   X = None  
   rho = None 

   if usecopula:
       G, X, rho = compute_copula_variables (rm, r)

   #print type(G), type(X), type(rho)
  
   for i in range (rating):
       cdf[i, 0] = pr[i, 0]
   
   for i in range(rating):
       for j in range(1,rating):
           cdf[i, j] = pr[i, j] + cdf[i, j-1]

   if setval != None:
        setval.setValue(0)
        setval.setLabelText("Monte Carlo simulation")
   
   for run in range(numofrun):

       tot = numpy.zeros((rating,tprev), dtype='float64')
       cont = numpy.zeros((rating,tprev), dtype='int')
       xi = numpy.random.rand(countries,tprev)
       x[:, 0] = rm[:, time-1]

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
                       cont[i, t] = cont[i, t] + 1
                       tot[i, t] = cont[i, t] * meanval[i]
               
           summa = 0.0
           for a in range(bp.shape[0]):
               summa += bp[a, t, run]
           r_prev[t, run] = summa
   
       for t in range(tprev):
           for i in range(rating):
                ac[i, t, run] = tot[i, t]/r_prev[t, run]
                if ac[i, t, run] != 0.0:
                    t1[t, run] += (ac[i, t, run]*tiv[i])
                    t2[t, run] += (ac[i, t, run]*math.log(float(rating)*ac[i, t, run]))
                    if cont[i, t] != 0:
                       term[t, run] += ac[i, t, run]* \
                               math.log(float(countries)/(float(rating)*cont[i, t]))
    
           entr[t, run] = t1[t, run] + t2[t, run] + term[t, run]
   
       if verbose:
           basicutils.progress_bar(run+1, numofrun)

       if setval != None:
           setval.setValue(100.0*(float(run+1)/float(numofrun)))
           if setval.wasCanceled():
             errmsg.append("Cancelled!")
             return False
   
   if verbose:
     print " "
   
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

def main_mkc_prop (rm, pr):

   countries = rm.shape[0]
   rating = numpy.max(rm)
   time = rm.shape[1]

   cdf = numpy.zeros((rating,rating), dtype='float64')

   for i in range (rating):
       cdf[i, 0] = pr[i, 0]
   
   for i in range(rating):
       for j in range(1,rating):
           cdf[i, j] = pr[i, j] + cdf[i, j-1]

   x = numpy.zeros((countries,time), dtype='int')
   xi = numpy.random.rand(countries,time)
   
   for c in range(countries):
       x[c, 0] = rm[c, 0]
   
   for c in range(countries):
       if xi[c, 0] <= cdf[x[c, 0]-1, 0]:
           x[c, 1] = 1
   
       for k in range(1,rating):
           if (cdf[x[c, 0]-1, k-1] < xi[c, 0]) and \
                   (xi[c, 0] <= cdf[x[c, 0]-1, k] ):
              x[c, 1] = k + 1
   
       for t in range(2,time):
           if xi[c, t-1] <= cdf[x[c, t-1]-1, 0]:
               x[c, t] = 1
   
           for k in range(1,rating):
               if (cdf[x[c, t-1]-1, k-1] < xi[c, t-1]) \
                       and (xi[c, t-1] <= cdf[x[c, t-1]-1, k]):
                 x[c, t] = k + 1
   
   return x

#####################################################################

def compute_copula_variables (ratings, spread):

    if ratings.shape != spread.shape:
        print "Error  in matrix dimension"
        exit(1)
    
    N = numpy.max(ratings)
    Nnaz = spread.shape[0]
    Dst = max(spread.shape)
    
    inc_spread = numpy.zeros((Nnaz,Dst-1))
    
    end = spread.shape[1]
    for i in range(Nnaz):
        a = spread[i,1:end] - spread[i,0:end-1]
        b = spread[i,0:end-1]
        inc_spread[i,:] = numpy.divide(a, b, out=numpy.full_like(a, 
            float("Inf")), where=b!=0)
    
    rttmp = ratings[:,1:end]
    totdim = rttmp.shape[0]*rttmp.shape[1]
    
    rttmp = rttmp.reshape(totdim, order='F')
    f_inc_spread = inc_spread.reshape(totdim, order='F')
    
    X = []
    G = []
    
    for i in range(N):
        tmp = numpy.where(rttmp == i+1)[0]
        dist_sp = [f_inc_spread[j] for j in tmp]
        dist_sp = filter(lambda a: a != float("Inf"), dist_sp)
        mind = scipy.stats.mstats.mquantiles(dist_sp, 0.05)
        maxd = scipy.stats.mstats.mquantiles(dist_sp, 0.95)
    
        dist_sp = filter(lambda a: a >= mind and a <= maxd, dist_sp)
    
        x, y = basicutils.ecdf(dist_sp)
        X.append(x)
        G.append(y)

    rho = numpy.corrcoef(spread) 

    return G, X, rho

#####################################################################


