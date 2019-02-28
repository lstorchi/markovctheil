import numpy
import math
import sys
import os

import scipy.stats

###############################################################################

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

###############################################################################

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

###############################################################################

def mat_to_stdout (mat):

    for i in range(mat.shape[0]):
       for j in range(mat.shape[1]):
           sys.stdout.write ("%f "%mat[i, j])
       sys.stdout.write ("\n")

###############################################################################

def vct_to_stdout (mat):

    for i in range(mat.shape[0]):
        sys.stdout.write ("%10.5f \n"%mat[i])

###############################################################################

def vct_to_file (bpm, oufilename):

    if os.path.exists(oufilename):
        os.remove(oufilename)
    
    outf = open(oufilename, "w")

    for i in range(bpm.shape[0]):
        outf.write(" %f "%(bpm[i]))
    outf.write("\n")
        
    outf.close()

###############################################################################

def mat_to_file (bpm, oufilename):

    if os.path.exists(oufilename):
        os.remove(oufilename)
    
    outf = open(oufilename, "w")

    for i in range(bpm.shape[0]):
        for j in range(bpm.shape[1]):
            outf.write(" %f "%(bpm[i, j]))
        outf.write("\n")
        
    outf.close()

###############################################################################

def histo_to_file (xh, h, oufilename):

    if os.path.exists(oufilename):
        os.remove(oufilename)
    
    outf = open(oufilename, "w")

    for i in range(len(h)):
        outf.write("%f %f\n"%((xh[i]+xh[i+1])/2.0, h[i]))
        
    outf.close()

###############################################################################

def progress_bar (count, total, status=''):

    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() 

###############################################################################

def get_histo (v, step):
    
    minh = min(v)
    maxh = max(v)
    nbins = int ((maxh - minh) / step)
    h, xh = numpy.histogram(v, bins=nbins, range=(minh, maxh))
    p = h/float(sum(h))

    t = sum((p*(math.log(len(p)))*p))

    bins = []
    for i in range(len(h)):
        bins.append((xh[i]+xh[i+1])/2.0) 

    return p, t, h, xh, nbins

###############################################################################

def extract_ti_mean (dat, step, rtnum, numofrun, fname = "", verbose = False):

    dat = dat[numpy.isfinite(dat)]
    pdat, tdat, hdat, xdat, nbins = get_histo(dat, step)

    if fname != "" and verbose:
      plt.hist(dat, normed=False, bins=nbins, facecolor='green')
      plt.xlabel(fname)
      plt.ylabel("f(x)")
      plt.title(fname)
      plt.grid(True)
      plt.savefig(fname+"_"+str(numofrun)+".eps")

    if fname != "":
      histo_to_file (xdat, hdat, fname+"_"+str(numofrun)+".txt")

    return dat, numpy.mean(dat), tdat, nbins, numpy.std(dat)

###############################################################################
# adapted from
# https://stackoverflow.com/questions/33345780/empirical-cdf-in-python-similiar-to-matlabs-one

def ecdf(raw_data):
    
    data = numpy.asarray(raw_data)
    data = numpy.atleast_1d(data)
    quantiles, counts = numpy.unique(data, return_counts=True)
    cumprob = numpy.cumsum(counts).astype(numpy.double) / data.size

    return quantiles, cumprob

#######################################################################
# adapted from 
# https://github.com/stochasticresearch/copula-py

def gaussian_copula_rnd (rho, m):
    
    n = rho.shape[0]
    mu = numpy.zeros(n)
    y = scipy.stats.multivariate_normal(mu,rho)
    mvndata = y.rvs(size = m)
    u = scipy.stats.norm.cdf(mvndata)

    return u

#######################################################################
