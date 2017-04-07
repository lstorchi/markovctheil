import numpy
import math
import sys
import os

###############################################################################

def mat_to_stdout (mat):

    for i in range(mat.shape[0]):
       for j in range(mat.shape[1]):
           sys.stdout.write ("%f "%mat[i][j])
       sys.stdout.write ("\n")

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
            outf.write(" %f "%(bpm[i][j]))
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
