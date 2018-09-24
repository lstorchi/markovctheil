import numpy
import math
import sys

import basicutils

###############################################################################

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

def compute_cps (rm, c_p1, performtest = False, c_p2 = -1, c_p3 = -1):

    countries=rm.shape[0]
    rating=numpy.max(rm)
    time=rm.shape[1]

    if (rating <= 0) or (rating > 8):
        raise Error ("rating " + rating + " is not a valid value")

    if c_p1 > 0:

        pr = 0
        pr1 = 0
        pr2 = 0
        pr3 = 0
        pr4 = 0

        if performtest:
            pr = numpy.zeros((rating,rating),dtype='float64')
            pr1 = numpy.zeros((rating,rating),dtype='float64')
            pr2 = numpy.zeros((rating,rating),dtype='float64')

            nk = numpy.zeros((rating,rating,countries), dtype='int64')
            num = numpy.zeros((rating,rating), dtype='int64')
            den = numpy.zeros(rating, dtype='int64')
            
            L = 0.0
            
            for i in range(rating):
                for j in range(rating):
            
                    for c  in range(countries):
                        for t in range(time-1):
                            if (rm[c, t] == (i+1)) and (rm[c, t+1] == (j+1)):
                                nk[i, j, c] = nk[i, j, c] + 1
                            
                    num[i, j] = sum(nk[i, j])
                
                den[i] = sum(num[i])
            
                if (den[i] > 0.0):
                    for j in range(rating):
                        val = numpy.float64(num[i,j])/numpy.float64(den[i])
                        if (val > 0.0):
                            L += num[i,j]*math.log(val) 
            
            for i in range(rating):
                for j in range(rating):
                    if den[i] != 0:
                       pr[i, j] = float(num[i, j])/float(den[i])
                    else: 
                       pr[i, j] = 0
                       pr[i, i] = 1
 

        num1 = numpy.zeros((rating,rating), dtype='int64')
        den1 = numpy.zeros(rating, dtype='int64')
        
        L1 = 0.0 
        
        for i in range(rating):
             for j in range(rating):
                for c in range(countries):
                     for t in range(c_p1-1):
                        if (rm[c, t] == (i+1)) and (rm[c, t+1] == (j+1)):
                            num1[i, j] = num1[i, j] + 1
                
             den1[i] = sum(num1[i])
        
             if (den1[i] > 0.0):
                for j in range(rating):
                   val = numpy.float64(num1[i,j])/numpy.float64(den1[i])
                   if (val > 0.0):
                      L1 += num1[i,j]*math.log(val) 
        
        if performtest:
             for i in range(rating):
                 for j in range(rating):
                     if den1[i] != 0:
                        pr1[i, j] = float(num1[i, j])/float(den1[i])
                     else: 
                        pr1[i, j] = 0        
                        pr1[i,i] = 1  

        num2 = numpy.zeros((rating,rating), dtype='int64')
        den2 = numpy.zeros(rating, dtype='int64')
         
        L2 = 0.0 
       
        if c_p2 > 0:

            if performtest:
                pr3 = numpy.zeros((rating,rating),dtype='float64')
           
            for i in range(rating):
                 for j in range(rating):
                     for c in range(countries):
                          for t in range(c_p1,c_p2-1) :
                              if (rm[c, t] == (i+1)) and (rm[c, t+1] == (j+1)):
                                  num2[i, j] = num2[i, j] + 1
                     
                 den2[i] = sum(num2[i])
            
                 if (den2[i] > 0.0):
                     for j in range(rating):
                        val = numpy.float64(num2[i,j])/numpy.float64(den2[i])
                        if (val > 0.0):
                            L2 += num2[i,j]*math.log(val) 

            if performtest:

                 for i in range(rating):
                     for j in range(rating):
                         if den2[i] != 0:
                            pr2[i, j] = float(num2[i, j])/float(den2[i])
                         else: 
                            pr2[i, j] = 0        
                            pr2[i,i] = 1  

            num3 = numpy.zeros((rating,rating), dtype='int64')
            den3 = numpy.zeros(rating, dtype='int64')
             
            L3 = 0.0 

            if c_p3 > 0:

                if performtest:
                    pr4 = numpy.zeros((rating,rating),dtype='float64')
           
                for i in range(rating):
                     for j in range(rating):
                         for c in range(countries):
                              for t in range(c_p2,c_p3-1) :
                                  if (rm[c, t] == (i+1)) and (rm[c, t+1] == (j+1)):
                                      num3[i, j] = num3[i, j] + 1
                         
                     den3[i] = sum(num3[i])
                
                     if (den3[i] > 0.0):
                         for j in range(rating):
                            val = numpy.float64(num3[i,j])/numpy.float64(den3[i])
                            if (val > 0.0):
                                L3 += num3[i,j]*math.log(val) 

                if performtest:

                    for i in range(rating):
                        for j in range(rating):
                            if den3[i] != 0:
                               pr3[i, j] = float(num3[i, j])/float(den3[i])
                            else: 
                               pr3[i, j] = 0
                               pr3[i,i] = 1
                
                num4 = numpy.zeros((rating,rating), dtype='int64')
                den4 = numpy.zeros(rating, dtype='int64')
                 
                L4 = 0.0 
                
                for i in range(rating):
                     for j in range(rating):
                         for c in range(countries):
                              for t in range(c_p3,time-1) :
                                  if (rm[c, t] == (i+1)) and (rm[c, t+1] == (j+1)):
                                      num4[i, j] = num4[i, j] + 1
                         
                     den4[i] = sum(num4[i])
                     if (den4[i] > 0.0):
                         for j in range(rating):
                            val = numpy.float64(num4[i,j])/numpy.float64(den4[i])
                            if (val > 0.0):
                                L4 += num4[i,j]*math.log(val) 

                if performtest:

                     for i in range(rating):
                         for j in range(rating):
                             if den4[i] != 0:
                                pr4[i, j] = float(num4[i, j])/float(den4[i])
                             else: 
                                pr4[i, j] = 0        
                                pr4[i,i] = 1  

                     return L, L1, L2, L3, L4, pr1, pr2, pr3, pr4

                
                return L1, L2, L3, L4

            else:
                for i in range(rating):
                     for j in range(rating):
                         for c in range(countries):
                              for t in range(c_p2,time-1) :
                                  if (rm[c, t] == (i+1)) and (rm[c, t+1] == (j+1)):
                                      num3[i, j] = num3[i, j] + 1
                         
                     den3[i] = sum(num3[i])
                
                     if (den3[i] > 0.0):
                         for j in range(rating):
                            val = numpy.float64(num3[i,j])/numpy.float64(den3[i])
                            if (val > 0.0):
                                L3 += num3[i,j]*math.log(val) 

                if performtest:

                    for i in range(rating):
                        for j in range(rating):
                            if den3[i] != 0:
                               pr3[i, j] = float(num3[i, j])/float(den3[i])
                            else: 
                               pr3[i, j] = 0
                               pr3[i,i] = 1

                    return L, L1, L2, L3, pr1, pr2, pr3
 
                return L1, L2, L3 

        else:

            for i in range(rating):
                 for j in range(rating):
                     for c in range(countries):
                          for t in range(c_p1,time-1) :
                              if (rm[c, t] == (i+1)) and (rm[c, t+1] == (j+1)):
                                  num2[i, j] = num2[i, j] + 1
                     
                 den2[i] = sum(num2[i])
            
                 if (den2[i] > 0.0):
                     for j in range(rating):
                        val = numpy.float64(num2[i,j])/numpy.float64(den2[i])
                        if (val > 0.0):
                            L2 += num2[i,j]*math.log(val) 

                 if performtest:

                     for i in range(rating):
                         for j in range(rating):
                             if den2[i] != 0:
                                pr2[i, j] = float(num2[i, j])/float(den2[i])
                             else: 
                                pr2[i, j] = 0
                                pr2[i,i] = 1

                     return L, L1, L2, pr1, pr2
 
 
            return L1, L2

    raise "at least cp1 should be > 0"

###############################################################################


