import numpy
import math
import sys

import basicutils

###############################################################################

def compute_cps (rm, errmsg, c_p1, c_p2 = -1, c_p3 = -1):

    countries=rm.shape[0]
    rating=numpy.max(rm)
    time=rm.shape[1]

    if (rating <= 0) or (rating > 8):
        errmsg.append("rating " + rating + " is not a valid value")
        return None, 

    if c_p1 > 0:
        num1 = numpy.zeros((rating,rating), dtype='int64')
        den1 = numpy.zeros(rating, dtype='int64')
        pr1 = numpy.zeros((rating,rating),dtype='float64')
        
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

        num2 = numpy.zeros((rating,rating), dtype='int64')
        den2 = numpy.zeros(rating, dtype='int64')
        pr2 = numpy.zeros((rating,rating),dtype='float64')
         
        L2 = 0.0 
        
        if c_p2 > 0:
           
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
            
            num3 = numpy.zeros((rating,rating), dtype='int64')
            den3 = numpy.zeros(rating, dtype='int64')
            pr3 = numpy.zeros((rating,rating),dtype='float64')
             
            L3 = 0.0 


            if c_p3 > 0:
            
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
                
                num4 = numpy.zeros((rating,rating), dtype='int64')
                den4 = numpy.zeros(rating, dtype='int64')
                pr4 = numpy.zeros((rating,rating),dtype='float64')
                 
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
                
                return L1, pr1, L2, pr2, L3, pr3, L4, pr4

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

                return L1, pr1, L2, pr2, L3, pr3 

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
 
            return L1, pr1, L2, pr2

    return None, 

###############################################################################


