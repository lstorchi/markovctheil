import numpy
import math
import sys

import time as tempo

import basicutils

###############################################################################

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

###############################################################################


class changepoint:

    def __init__(self):

        self.__metacommunity__ = None # ms
        self.__num_of_bootstrap_iter__ = 100 # num_of_run
        
        self.__cp_fortest__ = False # cp_fortest
        self.__cp_fortest_2__ = False # cp_fortest_2
        self.__cp_fortest_3__ = False # cp_fortest_3

        self.__num_of_cps__ = 1 # anumofcp
        self.__cp1_start__ = -1 # acp1start
        self.__cp2_start__ = -1 # acp2start
        self.__cp3_start__ = -1 # acp3start

        self.__cp1_stop__ = -1 # acp1stop
        self.__cp2_stop__ = -1 # acp2stop
        self.__cp3_stop__ = -1 # acp3stop
        
        self.__delta_cp__ = 1 # adeltacp

        self.__print_iter_info__ = False # aiterations

        self.__verbose__ = False # verbose 
        self.__fp__ = None # fp

    def set_metacommunity(self, inmat):
        if not isinstance(inmat, numpy.ndarray):
            raise TypeError("input must be a numpy array")

        self.__metacommunity__ = inmat


    def get_metacommunity(self):

        return self.__metacommunity__

    det set_num_of_bootstrap_iter (self, inval):
        if not isinstance(inval, int):
            raise TypeError("input must be an integer")

        self.__num_of_bootstrap_iter__ = inval


    def get_num_of_bootstrap_iter (self):

        return self.__num_of_bootstrap_iter__
        

    def set_cp1_fortest (self, inval):
        if not isinstance(inval, bool):
            raise TypeError("input must be a boolean")

        self.__cp_fortest_1__ = inval


    def get_cp1_fortest (self):

        return self.__cp_fortest_1__


    def set_cp2_fortest (self, inval):
        if not isinstance(inval, bool):
            raise TypeError("input must be a boolean")

        self.__cp_fortest_2__ = inval


    def get_cp2_fortest (self):

        return self.__cp_fortest_2__


    def set_cp3_fortest (self, inval):
        if not isinstance(inval, bool):
            raise TypeError("input must be a boolean")

        self.__cp_fortest_3__ = inval


    def get_cp3_fortest (self):

        return self.__cp_fortest_3__


    def set_num_of_cps (self, inval):
        if not isinstance(inval, int):
            raise TypeError("input must be an integer")

        self.__num_of_cps__ = inval


    def get_num_of_cps (self):

        return self.__num_of_cps__

    def set_cp1_start_stop (self, start=-1, stop=-1):
        if not isinstance(start, int):
            raise TypeError("input must be an integer")

        if not isinstance(stop, int):
            raise TypeError("input must be an integer")

        self.__cp1_start__ = start
        self.__cp1_stop__ = stop


    def get_cp1_start_stop (self):

        return self.__cp1_start__, self.__cp1_stop__


    def set_cp2_start_stop (self, start=-1, stop=-1):
        if not isinstance(start, int):
            raise TypeError("input must be an integer")

        if not isinstance(stop, int):
            raise TypeError("input must be an integer")

        self.__cp2_start__ = start
        self.__cp2_stop__ = stop


    def get_cp2_start_stop (self):

        return self.__cp2_start__, self.__cp2_stop__


    def set_cp3_start_stop (self, start=-1, stop=-1):
        if not isinstance(start, int):
            raise TypeError("input must be an integer")

        if not isinstance(stop, int):
            raise TypeError("input must be an integer")

        self.__cp3_start__ = start
        self.__cp3_stop__ = stop


    def get_cp3_start_stop (self):

        return self.__cp3_start__, self.__cp3_stop__


    def set_delta_cp (self, inval):
        if not isinstance(inval, int):
            raise TypeError("input must be an integer")

        self.__delta_cp__ = inval


    def get_delta_cp (self):

        return self.__delta_cp__

    
    def set_file_pointer (self, fpin):
        if not isinstance(fpin, io.IOBase):
            raise TypeError("input must be a file pointer")

        self.__fp__ = fpin


    def get_file_pointer (self):
        
        return self.__fp__


    def set_print_iter_info (self, inval):
        if not isinstance(fpin, bool):
            raise TypeError("input must be a bool")

        self.__print_iter_info__ = inval


    def get_print_iter_info (self):

        return self.__print_iter_info__


    def set_verbose (self, inval):
        if not isinstance(fpin, bool):
            raise TypeError("input must be a bool")

        self.__verbose__ = inval

    def get_verbose (self):

        return self.__verbose__
    
    def compute_cps (self, ms, num_of_run, cp_fortest, cp_fortest_2, cp_fortest_3, 
        anumofcp, acp1start, acp2start, acp3start, acp1stop, acp2stop, 
        acp3stop, adeltacp, aiterations, fp = None, verbose = False,
        setval=None):

    if setval != None:
        setval.setValue(0)
        setval.setLabelText("ChangePoint analysis")

    rating = numpy.max(ms)
    time = ms.shape[1]

    if cp_fortest >= 0 and num_of_run >= 0:
    
        if (verbose):
            print "Startint CP test..."
    
        L = 0.0
        L1 = 0.0
        L2 = 0.0
        L3 = 0.0
        L4 = 0.0
    
        pr1 = 0.0
        lambdastart = 0.0
    
        if anumofcp == 1:
            L, L1, L2, pr1, pr2 = self.__compute_cps__ (ms, cp_fortest, True)
            lambdastart = 2.0*((L1+L2)-L)
        elif anumofcp == 2:
            L, L1, L2, L3, pr1, pr2, pr3 = \
            self.__compute_cps__ (ms, cp_fortest, True, cp_fortest_2)
            lambdastart = 2.0*((L1+L2+L3)-L)
            if (verbose):
                print L, L1, L2, L3
        elif anumofcp == 3:
            L, L1, L2, L3, L4, pr1, pr2, pr3, pr4 = \
                    self.__compute_cps__ (ms, cp_fortest, True, \
                    cp_fortest_2, cp_fortest_3)
            lambdastart = 2.0*((L1+L2+L3+L4)-L)
            if (verbose):
                print L, L1, L2, L3, L4
    
        lambdas = []
    
        if (verbose):
            print "Starting iterations..."
    
        for i in range(num_of_run):
    
            start = tempo.time()
            cstart = tempo.clock()
     
            x = mainmkvcmp.main_mkc_prop (ms, pr1)
    
            lambdav = 0.0
    
            L, L1, L2, pr1_o, pr2_o = self.__compute_cps__ (x, cp_fortest, True)
    
            if anumofcp == 1:
                L, L1, L2, pr1_o, pr2_o = self.__compute_cps__ (x, cp_fortest, True)
                lambdav = 2.0*((L1+L2)-L)
            elif anumofcp == 2:
                L, L1, L2, L3, pr1_o, pr2_o, pr3_o = \
                self.__compute_cps__ (x, cp_fortest, True, cp_fortest_2)
                lambdav = 2.0*((L1+L2+L3)-L)
            elif anumofcp == 3:
                L, L1, L2, L3, L4, pr1_o, pr2_o, pr3_o, pr4_o = \
                        self.__compute_cps__ (x, cp_fortest, True, \
                        cp_fortest_2, cp_fortest_3)
                lambdav = 2.0*((L1+L2+L3+L4)-L)
    
            lambdas.append(lambdav) 
    
            if not (fp is None):
                fp.write(str(i+1) + " " + str(lambdav) + "\n") 
    
            end = tempo.time()
            cend = tempo.clock()
        
            if (verbose):
                print "%10d of %10d time (%10.5f s CPU time %10.5f s)"%(i+1 , num_of_run, 
                     end - start, cend - cstart)
     
    
        idx95 = int(num_of_run*0.95+0.5)
    
        if idx95 >= num_of_run:
            idx95 = num_of_run - 1
    
        lamda95 = lambdas[idx95]
    
        minrat = numpy.min(ms)
        maxrat = numpy.max(ms)
    
        #ndof = (maxrat - minrat + 1) * (maxrat - minrat)
        #ndof = max (positive1, positive2)
    
        #chi2 = scipy.stats.chi2.isf(0.05, ndof)
        #pvalue = 1.0 - scipy.stats.chi2.cdf(lamda95, ndof)
    
        pvalue = (1.0 / numpy.float64(num_of_run + 1)) * \
                (1.0 + numpy.float64(sum(i >= lambdastart for i in lambdas)))
    
        if (verbose):
            #print "Ndof        : ", ndof
            print "Lambda(95%) : ", lamda95
            print "Lambda      : ", lambdastart
            print "P-Value     : ", pvalue

        vals = [lamda95, lambdastart, pvalue]
        return vals
    
    else:
    
        maxval = -1.0 * float("inf")
    
        if (anumofcp == 1):
        
            cp1stop = time-1
        
            if acp1start <= 0 or acp1start > time-1:
                raise Error ("CP1 start invalid value")
        
            if acp1stop < 0:
                cp1stop = time-1
            else:
                cp1stop = acp1stop
        
            if cp1stop <= acp1start or cp1stop > time-1:
                raise Error ("CP1 stop invalid value")
        
            cp = 0
            idx = 0
            allvalues = []
            for c_p in range(acp1start, cp1stop):
                start = tempo.time()
                cstart = tempo.clock()

        
                try:
                    L1, L2 = self.__compute_cps__ (ms, c_p)
                except Error:
                    raise Error ("Oops! error in the main function") 
        
                if (maxval < L1+L2):
                    maxval = L1 + L2
                    cp = c_p
            
                if not (fp is None):
                    fp.write(str(c_p) + " " + str(L1+L2) + "\n")
                allvalues.append((c_p, L1+L2))
        
                end = tempo.time()
                cend = tempo.clock()
        
                if (verbose):
                    if aiterations:
                        print "%10d of %10d time (%10.5f s CPU time %10.5f s)"%(idx+1 , cp1stop-acp1start, 
                            end - start, cend - cstart)
                    else:
                        basicutils.progress_bar(idx+1, cp1stop-acp1start)

                if not(setval is None):
                    setval.setValue(100.0*(float(idx+1)/float(cp1stop-acp1start)))
                    if setval.wasCanceled():
                        raise Error("Cancelled")

                idx = idx + 1 
        
            if (verbose):
                print ""
                print ""
                print "Change Point: ", cp, " (",maxval, ")"
            
            vals = [cp, maxval, allvalues]
            return vals

        elif (anumofcp == 2):
            cp1 = 0
            cp2 = 0
        
            if adeltacp > 1:
        
               cp1stop = time-1
        
               if acp1start <= 0 or acp1start > time-1:
                   raise ("CP1 start invalid value")
              
               if acp1stop < 0:
                   cp1stop = time-1
               else:
                   cp1stop = acp1stop
              
               if cp1stop <= acp1start or cp1stop > time-1:
                   raise Error ("CP1 stop invalid value")
        
               # I am lazy 
               tot = 0
               for c_p1 in range(acp1start, cp1stop):
                   for c_p2 in range(c_p1 + adeltacp, time-1):
                       tot = tot + 1
        
               idx = 0
               allvalues = []
               for c_p1 in range(acp1start, cp1stop):
                   for c_p2 in range(c_p1 + adeltacp, time-1):
                       start = tempo.time()
                       cstart = tempo.clock()
        
                       try:
                           L1, L2, L3 = self.__compute_cps__ (ms, c_p1, False, c_p2)
                       except Error:
                           raise Error ("Oops! error in the main function")
         
                       if (maxval < L1+L2+L3):
                           maxval = L1 + L2 + L3
                           cp1 = c_p1
                           cp2 = c_p2
        
                       end = tempo.time()
                       cend = tempo.clock()
        
                       if (verbose):
                            if aiterations:
                                print "%10d of %10d time (%10.5f s CPU time %10.5f s)"%(idx+1 , tot, 
                                    end - start, cend - cstart)
                            else:
                                basicutils.progress_bar(idx+1, tot)

                       if not(setval is None):
                           setval.setValue(100.0*(float(idx+1)/float(tot)))
                           if setval.wasCanceled():
                               raise Error("Cancelled")

                       idx = idx + 1 
         
                       if not (fp is None):
                          fp.write(str(c_p1) + " " + str(c_p2) + " " 
                               + str(L1+L2+L3) + "\n")

                       allvalues.append((c_p1, c_p2, L1+L2+L3))

               if (verbose):
                   print ""
                   print ""
                   print "Change Point: ", cp1, " , ", cp2, " (",maxval, ")"
               
               vals = [cp1, cp2, maxval, allvalues]
               return vals

            else:
        
               cp1stop = time-1
        
               if acp1start <= 0 or acp1start > time-1:
                   raise Error ("CP1 start invalid value")
              
               if acp1stop < 0:
                   cp1stop = time-1
               else:
                   cp1stop = acp1stop
              
               if cp1stop <= acp1start or cp1stop > time-1:
                   raise Error ("CP1 stop invalid value")
        
               cp2stop = time-1
        
               if acp2start <= 0 or acp2start > time-1:
                   raise Error ("CP2 start invalid value")
              
               if acp2stop < 0:
                   cp2stop = time-1
               else:
                   cp2stop = acp2stop
              
               if cp2stop <= acp2start or cp2stop > time-1:
                   raise Error ("CP2 stop invalid value")
        
               if acp2start <= acp1start:
                   raise Error ("CP2 start invalid value")
         
               # I am lazy
               tot = 0
               for c_p1 in range(acp1start, cp1stop):
                   for c_p2 in range(acp2start, cp2stop):
                       tot = tot + 1

               idx = 0
               allvalues = []
               for c_p1 in range(acp1start, cp1stop):
                   for c_p2 in range(acp2start, cp2stop):
        
                       start = tempo.time()
                       cstart = tempo.clock()
        
                       try:
                           L1, L2, L3 = self.__compute_cps__ (ms, c_p1, False, c_p2)
                       except Error:
                           raise Error ("Oops! error in the main function") 
               
                       if (maxval < L1+L2+L3):
                           maxval = L1 + L2 + L3
                           cp1 = c_p1
                           cp2 = c_p2
        
                       end = tempo.time()
                       cend = tempo.clock()
        
                       if (verbose):
                            if aiterations:
                                print "%10d of %10d time (%10.5f s CPU time %10.5f s)"%(idx+1 , tot, 
                                    end - start, cend - cstart)
                            else:
                                basicutils.progress_bar(idx+1, tot)

                       if not(setval is None):
                           setval.setValue(100.0*(float(idx+1)/float(tot)))
                           if setval.wasCanceled():
                               raise Error("Cancelled")
        
                       idx = idx + 1 
         
                       if not (fp is None):
                          fp.write(str(c_p1) + " " + str(c_p2) + " " 
                               + str(L1+L2+L3) + "\n")

                       allvalues.append((c_p1, c_p2, L1+L2+L3))

               if (verbose):
                    print ""
                    print ""
                    print "Change Point: ", cp1, " , ", cp2 ," (",maxval, ")"

               vals = [cp1, cp2, maxval, allvalues]
               return vals

        elif (anumofcp == 3):
            cp1 = 0
            cp2 = 0
            cp3 = 0
        
            if adeltacp > 1:
        
               cp1stop = time-1
        
               if acp1start <= 0 or acp1start > time-1:
                   raise Error ("CP1 start invalid value")
              
               if acp1stop < 0:
                   cp1stop = time-1
               else:
                   cp1stop = acp1stop
              
               if cp1stop <= acp1start or cp1stop > time-1:
                   raise Error ("CP1 stop invalid value")
        
               # I am lazy
               tot = 0
               for c_p1 in range(acp1start, cp1stop):
                   for c_p2 in range(c_p1 + adeltacp, time-1):
                       for c_p3 in range(c_p2 + adeltacp, time-1):
                           tot = tot + 1
        
               idx = 0
               allvalues = []
               for c_p1 in range(acp1start, cp1stop):
                   for c_p2 in range(c_p1 + adeltacp, time-1):
                       for c_p3 in range(c_p2 + adeltacp, time-1):
        
                           start = tempo.time()
                           cstart = tempo.clock()
        
                           try:
                               L1, L2, L3, L4 = self.__compute_cps__ (ms, c_p1, False, c_p2, c_p3)
                           except Error:
                               raise Error ("Oops! error in the main function") 
                           
                           if (maxval < L1+L2+L3+L4):
                               maxval = L1 + L2 + L3 + L4
                               cp1 = c_p1
                               cp2 = c_p2
                               cp3 = c_p3
        
                           end = tempo.time()
                           cend = tempo.clock()
              
                           if (verbose):
                                if aiterations:
                                    print "%10d of %10d time (%10.5f s CPU time %10.5f s)"%(idx+1 , tot, 
                                            end - start, cend - cstart)
                                else:
                                    basicutils.progress_bar(idx+1, tot)

                           if not(setval is None):
                               setval.setValue(100.0*(float(idx+1)/float(tot)))
                               if setval.wasCanceled():
                                  raise Error("Cancelled")
                            
                           idx = idx + 1 
                          
                           if not (fp is None):
                              fp.write(str(c_p1) + " " + str(c_p2) + " " 
                                   + str(c_p3) + " " 
                                   + str(L1+L2+L3+L4) + "\n")
                           allvalues.append((c_p1, c_p2, c_p3, L1+L2+L3+L4))

               if (verbose):
                    print ""
                    print ""
                    print "Change Point: ", cp1, " , ", cp2, " ", cp3, " (",maxval, ")"

               vals = [cp1, cp2, cp3, maxval, allvalues]
               return vals

            else:
        
               cp1stop = time-1
        
               if acp1start <= 0 or acp1start > time-1:
                   raise Error ("CP1 start invalid value")
              
               if acp1stop < 0:
                   cp1stop = time-1
               else:
                   cp1stop = acp1stop
              
               if cp1stop <= acp1start or cp1stop > time-1:
                   raise Error ("CP1 stop invalid value")
        
               cp2stop = time-1
        
               if acp2start <= 0 or acp2start > time-1:
                   raise Error ("CP2 start invalid value")
              
               if acp2stop < 0:
                   cp2stop = time-1
               else:
                   cp2stop = acp2stop
              
               if cp2stop <= acp2start or cp2stop > time-1:
                   raise Error ("CP2 stop invalid value")
        
               if acp2start <= acp1start:
                   raise Error ("CP2 start invalid value")
        
               cp3stop = time-1
        
               if acp3start <= 0 or acp3start > time-1:
                   raise Error ("CP3 start invalid value")
              
               if acp3stop < 0:
                   cp3stop = time-1
               else:
                   cp3stop = acp3stop
              
               if cp3stop <= acp3start or cp3stop > time-1:
                   raise Error ("CP3 stop invalid value")
        
               if acp3start <= acp2start:
                   raise Error ("CP3 start invalid value")
         
               tot = 0
               for c_p1 in range(acp1start, cp1stop):
                   for c_p2 in range(acp2start, cp2stop):
                       for c_p3 in range(acp3start, cp3stop):
                           tot = tot + 1
        
               idx = 0
               allvalues = []
               for c_p1 in range(acp1start, cp1stop):
                   for c_p2 in range(acp2start, cp2stop):
                       for c_p3 in range(acp3start, cp3stop):
        
                           start = tempo.time()
                           cstart = tempo.clock()
        
                           try:
                               L1, L2, L3, L4 = self.__compute_cps__ (ms,  
                                   c_p1, False, c_p2, c_p3)
                           except Error:
                               raise Error ("Oops! error in the main function") 
         
                           if (maxval < L1+L2+L3+L4):
                               maxval = L1 + L2 + L3 + L4
                               cp1 = c_p1
                               cp2 = c_p2
                               cp3 = c_p3
        
                           end = tempo.time()
                           cend = tempo.clock()
              

                           if (verbose):
                                if aiterations:
                                    print "%10d of %10d time (%10.5f s CPU time %10.5f s)"%(idx+1 , tot, 
                                            end - start, cend - cstart)
                                else:
                                    basicutils.progress_bar(idx+1, tot)

                           if not(setval is None):
                               setval.setValue(100.0*(float(idx+1)/float(tot)))
                               if setval.wasCanceled():
                                  raise Error("Cancelled")
 
                           
                           idx = idx + 1 
                           if not (fp is None):
                                fp.write(str(c_p1) + " " + str(c_p2) + " " 
                                   + str(c_p3) + " " 
                                   + str(L1+L2+L3+L4) + "\n")
                           allvalues.append((c_p1, c_p2, c_p3, L1+L2+L3+L4))

               if (verbose):
                    print ""
                    print ""
                    print "Change Point: ", cp1, " , ", cp2 , " ", cp3, " (",maxval, ")"
               
               vals = [cp1, cp2, cp3, maxval, allvalues]
               return vals

    raise Error ("Nothing todo")


###############################################################################
# PRIVATE
###############################################################################

    def __compute_cps__ (slef, rm, c_p1, performtest = False, \
            c_p2 = -1, c_p3 = -1):

        countries=rm.shape[0]
        rating=numpy.max(rm)
        time=rm.shape[1]
        
        if (rating <= 0) or (rating > 8):
            raise Error ("rating " + rating + \
                    " is not a valid value")
        
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
                                      if (rm[c, t] == (i+1)) and (rm[c, t+1] == \
                                              (j+1)):
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
                                      if (rm[c, t] == (i+1)) and (rm[c, t+1] \
                                              == (j+1)):
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
                                      if (rm[c, t] == (i+1)) and (rm[c, t+1] \
                                              == (j+1)):
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
                                  if (rm[c, t] == (i+1)) and (rm[c, t+1] \
                                          == (j+1)):
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
        
        raise Error ("at least cp1 should be > 0")

