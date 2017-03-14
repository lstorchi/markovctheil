import numpy.linalg
import numpy.random
import scipy.stats
import scipy.io
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

def progress_bar (count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() 

###############################################################################
def get_osserv (ms, rating, countries, time):

    Nk = numpy.zeros((rating,rating,countries), dtype='int64')
    Num = numpy.zeros((rating,rating), dtype='int64')
    Den = numpy.zeros(rating, dtype='int64')
    Pr = numpy.zeros((rating,rating), dtype='float64')
    
    for k in range(countries):
        for t in range(time-1):
            for i in range(rating):
                for j in range(rating):
                    if (ms[k][t] == (i+1)) and (ms[k][t+1] == (j+1)):
                        Nk[i][j][k] = Nk[i][j][k] + 1
    
                    Num[i][j] = sum(Nk[i][j])
    
                Den[i] = sum(Num[i])
    
        progress_bar(k+1, countries)
    
    for i in range(rating):
        for j in range(rating):
            if Den[i] != 0:
                Pr[i][j] = float(Num[i][j])/float(Den[i])
            else: 
                Pr[i][j] = 0.0

    return Pr, Num, Den

###############################################################################

filename1 = "ms1.mat"
filename2 = "ms2.mat"

namems1 = 'ms1'
namems2 = 'ms2'

if len(sys.argv) != 5:
    print "usage: ", sys.argv[0], " ms2matfilename ms3matfilename mat1 mat2" 
    exit(1)
else:
    filename1 = sys.argv[1] 
    filename2 = sys.argv[2]
    namems1 = sys.argv[3]
    namems2 = sys.argv[4]

ms1d = scipy.io.loadmat(filename1)
ms2d = scipy.io.loadmat(filename2)

if not(namems1 in ms1d.keys()):
    print "Cannot find ", namems1, " in ", filename1
    print ms1d.keys()
    exit(1)

if not(namems2 in ms2d.keys()):
    print "Cannot find ", namems2, " in ", filename2
    print ms2d.keys()
    exit(1)

rating1 = numpy.max(ms1d[namems1])
rating2 = numpy.max(ms2d[namems2])

if (rating1 <= 0) or (rating1 > 8):
    print "rating ", rating1, " is not a valid value"
    exit(1)

if (rating2 <= 0) or (rating2 > 8):
    print "rating ", rating2, " is not a valid value"
    exit(1)

ms1 = ms1d[namems1]
ms2 = ms2d[namems2]
time1 = len(ms1[1,:])
time2 = len(ms2[1,:])

countries1 = ms1d[namems1].shape[0]
countries2 = ms2d[namems2].shape[0]

Pr1, Num1, Den1 = get_osserv (ms1, rating1, countries1, time1)
Pr2, Num2, Den2 = get_osserv (ms2, rating2, countries2, time2)

