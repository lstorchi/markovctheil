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

        self.__metacommunity__ = None 
        self.__num_of_bootstrap_iter__ = 100 
        
        self.__cp_fortest_1__ = -1
        self.__cp_fortest_2__ = -1
        self.__cp_fortest_3__ = -1

        self.__num_of_cps__ = 1 

        self.__cp1_start__ = -1 
        self.__cp2_start__ = -1 
        self.__cp3_start__ = -1 

        self.__cp1_stop__ = -1 
        self.__cp2_stop__ = -1 
        self.__cp3_stop__ = -1 
        
        self.__delta_cp__ = 1 

        self.__print_iter_info__ = False 

        self.__verbose__ = False 
        self.__fp__ = None 

        self.__cp1_found__ = -1
        self.__cp2_found__ = -1
        self.__cp3_found__ = -1
        self.__maxval__ = -float("inf")
        self.__allvalues__ = []

        self.__lambda95__ = None
        self.__lambdastart__ = None
        self.__pvalue__ = None

    def get_lambda95(self):
        return self.__lambda95__

    def get_lambdastart(self):
        return self.__lambdastart__

    def get_pvalue(self):
        return self.__pvalue__

    def get_cp1_found(self):
        return self.__cp1_found__

    def get_cp2_found(self):
        return self.__cp2_found__

    def get_cp3_found(self):
        return self.__cp3_found__

    def get_maxval(self):
        return self.__maxval__

    def get_allvalues(self):
        return self.__allvalues__

    def set_metacommunity(self, inmat):
        if not isinstance(inmat, numpy.ndarray):
            raise TypeError("set_metacommunity: input must be a numpy array")

        self.__metacommunity__ = inmat


    def get_metacommunity(self):

        return self.__metacommunity__

    def set_num_of_bootstrap_iter (self, inval):
        if not isinstance(inval, int):
            raise TypeError("input must be an integer")

        self.__num_of_bootstrap_iter__ = inval


    def get_num_of_bootstrap_iter (self):

        return self.__num_of_bootstrap_iter__
        

    def set_cp1_fortest (self, inval):
        if not isinstance(inval, int):
            raise TypeError("input must be an integer")

        self.__cp_fortest_1__ = inval


    def get_cp1_fortest (self):

        return self.__cp_fortest_1__


    def set_cp2_fortest (self, inval):
        if not isinstance(inval, int):
            raise TypeError("input must be an integer")

        self.__cp_fortest_2__ = inval


    def get_cp2_fortest (self):

        return self.__cp_fortest_2__


    def set_cp3_fortest (self, inval):
        if not isinstance(inval, int):
            raise TypeError("input must be an integer")

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
        if not hasattr(fpin, 'write'):
            raise TypeError("input has not write attribute")

        self.__fp__ = fpin


    def get_file_pointer (self):
        
        return self.__fp__


    def set_print_iter_info (self, inval):
        if not isinstance(inval, bool):
            raise TypeError("input must be a bool")

        self.__print_iter_info__ = inval


    def get_print_iter_info (self):

        return self.__print_iter_info__


    def set_verbose (self, inval):
        if not isinstance(inval, bool):
            raise TypeError("input must be a bool")

        self.__verbose__ = inval

    def get_verbose (self):

        return self.__verbose__
    
    def compute_cps (self, setval=None):

        if setval != None:
            setval.setValue(0)
            setval.setLabelText("ChangePoint analysis")
        
        rating = numpy.max(self.__metacommunity__)
        time = self.__metacommunity__.shape[1]
        
        if self.__cp_fortest_1__ >= 0 and self.__num_of_bootstrap_iter__ >= 0:
        
            if (self.__verbose__):
                print "Startint CP test..."
        
            L = 0.0
            L1 = 0.0
            L2 = 0.0
            L3 = 0.0
            L4 = 0.0
        
            pr1 = 0.0
            self.__lambdastart__ = 0.0
        
            if self.__num_of_cps__ == 1:
                L, L1, L2, pr1, pr2 = self.__compute_cps__ (\
                        self.__metacommunity__, self.__cp_fortest_1__, True)
                self.__lambdastart__ = 2.0*((L1+L2)-L)
            elif self.__num_of_cps__ == 2:
                L, L1, L2, L3, pr1, pr2, pr3 = \
                self.__compute_cps__ (self.__metacommunity__, \
                self.__cp_fortest_1__, True, self.__cp_fortest_2__)
                self.__lambdastart__ = 2.0*((L1+L2+L3)-L)
                if (self.__verbose__):
                    print L, L1, L2, L3
            elif self.__num_of_cps__ == 3:
                L, L1, L2, L3, L4, pr1, pr2, pr3, pr4 = \
                        self.__compute_cps__ (self.__metacommunity__, \
                        self.__cp_fortest_1__, True, \
                        elf.__cp_fortest_2__, self.__cp_fortest_3__)
                self.__lambdastart__ = 2.0*((L1+L2+L3+L4)-L)
                if (self.__verbose__):
                    print L, L1, L2, L3, L4
        
            lambdas = []
        
            if (self.__verbose__):
                print "Starting iterations..."
        
            for i in range(self.__num_of_bootstrap_iter__):
        
                start = tempo.time()
                cstart = tempo.clock()
         
                x = self.__mkc_prop__ (self.__metacommunity__, pr1)
        
                lambdav = 0.0
        
                L, L1, L2, pr1_o, pr2_o = self.__compute_cps__ (x, \
                        self.__cp_fortest_1__, True)
        
                if self.__num_of_cps__ == 1:
                    L, L1, L2, pr1_o, pr2_o = self.__compute_cps__ (x, \
                            self.__cp_fortest_1__, True)
                    lambdav = 2.0*((L1+L2)-L)
                elif self.__num_of_cps__ == 2:
                    L, L1, L2, L3, pr1_o, pr2_o, pr3_o = \
                    self.__compute_cps__ (x, \
                    self.__cp_fortest_1__, True, self.__cp_fortest_2__)
                    lambdav = 2.0*((L1+L2+L3)-L)
                elif self.__num_of_cps__ == 3:
                    L, L1, L2, L3, L4, pr1_o, pr2_o, pr3_o, pr4_o = \
                            self.__compute_cps__ (x, self.__cp_fortest_1__, True, \
                            self.__cp_fortest_2__, self.__cp_fortest_3__)
                    lambdav = 2.0*((L1+L2+L3+L4)-L)
        
                lambdas.append(lambdav) 
        
                if not (self.__fp__ is None):
                    self.__fp__.write(str(i+1) + " " + str(lambdav) + "\n") 
        
                end = tempo.time()
                cend = tempo.clock()
            
                if (self.__verbose__):
                    print "%10d of %10d time (%10.5f s CPU time %10.5f s)"%(\
                            i+1 , self.__num_of_bootstrap_iter__, 
                         end - start, cend - cstart)
         
        
            idx95 = int(self.__num_of_bootstrap_iter__*0.95+0.5)
        
            if idx95 >= self.__num_of_bootstrap_iter__:
                idx95 = self.__num_of_bootstrap_iter__ - 1
        
            self.__lambda95__ = lambdas[idx95]
        
            minrat = numpy.min(self.__metacommunity__)
            maxrat = numpy.max(self.__metacommunity__)
        
            #ndof = (maxrat - minrat + 1) * (maxrat - minrat)
            #ndof = max (positive1, positive2)
        
            #chi2 = scipy.stats.chi2.isf(0.05, ndof)
            #pvalue = 1.0 - scipy.stats.chi2.cdf(self.__lambda95__, ndof)
        
            self.__pvalue__ = \
                    (1.0 / numpy.float64(self.__num_of_bootstrap_iter__ + 1)) * \
                    (1.0 + numpy.float64(sum(i >= self.__lambdastart__ for i in lambdas)))
        
            if (self.__verbose__):
                #print "Ndof        : ", ndof
                print "Lambda(95%) : ", self.__lambda95__
                print "Lambda      : ", self.__lambdastart__
                print "P-Value     : ", self.__pvalue__
        
            return 
        
        else:
        
            maxval = -1.0 * float("inf")
        
            if (self.__num_of_cps__ == 1):
            
                cp1stop = time-1
            
                if self.__cp1_start__ <= 0 or self.__cp1_start__ > time-1:
                    raise Error ("CP1 start invalid value")
            
                if self.__cp1_stop__ < 0:
                    cp1stop = time-1
                else:
                    cp1stop = self.__cp1_stop__
            
                if cp1stop <= self.__cp1_start__ or cp1stop > time-1:
                    raise Error ("CP1 stop invalid value")
            
                cp = 0
                idx = 0
                self.__allvalues__ = []
                for c_p in range(self.__cp1_start__, cp1stop):
                    start = tempo.time()
                    cstart = tempo.clock()
        
            
                    try:
                        L1, L2 = self.__compute_cps__ (self.__metacommunity__, c_p)
                    except Error:
                        raise Error ("Oops! error in the main function") 
            
                    if (maxval < L1+L2):
                        maxval = L1 + L2
                        cp = c_p
                
                    if not (self.__fp__ is None):
                        self.__fp__.write(str(c_p) + " " + str(L1+L2) + "\n")
                    self.__allvalues__.append((c_p, L1+L2))
            
                    end = tempo.time()
                    cend = tempo.clock()
            
                    if (self.__verbose__):
                        if self.__print_iter_info__:
                            print "%10d of %10d time (%10.5f s CPU time %10.5f s)"%(\
                                    idx+1 , cp1stop-self.__cp1_start__, \
                                    end - start, cend - cstart)
                        else:
                            basicutils.progress_bar(idx+1, cp1stop-self.__cp1_start__)
        
                    if not(setval is None):
                        setval.setValue(100.0*(float(idx+1)/float(cp1stop-self.__cp1_start__)))
                        if setval.wasCanceled():
                            raise Error("Cancelled")
        
                    idx = idx + 1 
            
                if (self.__verbose__):
                    print ""
                    print ""
                    print "Change Point: ", cp, " (",maxval, ")"
                
                self.__cp1_found__ = cp
                self.__maxval__ = maxval

                return 
        
            elif (self.__num_of_cps__ == 2):
                cp1 = 0
                cp2 = 0
            
                if self.__delta_cp__ > 1:
            
                   cp1stop = time-1
            
                   if self.__cp1_start__ <= 0 or self.__cp1_start__ > time-1:
                       raise ("CP1 start invalid value")
                  
                   if self.__cp1_stop__ < 0:
                       cp1stop = time-1
                   else:
                       cp1stop = self.__cp1_stop__
                  
                   if cp1stop <= self.__cp1_start__ or cp1stop > time-1:
                       raise Error ("CP1 stop invalid value")
            
                   # I am lazy 
                   tot = 0
                   for c_p1 in range(self.__cp1_start__, cp1stop):
                       for c_p2 in range(c_p1 + self.__delta_cp__, time-1):
                           tot = tot + 1
            
                   idx = 0
                   self.__allvalues__ = []
                   for c_p1 in range(self.__cp1_start__, cp1stop):
                       for c_p2 in range(c_p1 + self.__delta_cp__, time-1):
                           start = tempo.time()
                           cstart = tempo.clock()
            
                           try:
                               L1, L2, L3 = self.__compute_cps__ (\
                                       self.__metacommunity__, c_p1, False, c_p2)
                           except Error:
                               raise Error ("Oops! error in the main function")
             
                           if (maxval < L1+L2+L3):
                               maxval = L1 + L2 + L3
                               cp1 = c_p1
                               cp2 = c_p2
            
                           end = tempo.time()
                           cend = tempo.clock()
            
                           if (self.__verbose__):
                                if self.__print_iter_info__:
                                    print "%10d of %10d time (%10.5f s CPU time %10.5f s)"%(\
                                            idx+1 , tot, end - start, cend - cstart)
                                else:
                                    basicutils.progress_bar(idx+1, tot)
        
                           if not(setval is None):
                               setval.setValue(100.0*(float(idx+1)/float(tot)))
                               if setval.wasCanceled():
                                   raise Error("Cancelled")
        
                           idx = idx + 1 
             
                           if not (self.__fp__ is None):
                              self.__fp__.write(str(c_p1) + " " + str(c_p2) + " " 
                                   + str(L1+L2+L3) + "\n")
        
                           self.__allvalues__.append((c_p1, c_p2, L1+L2+L3))
        
                   if (self.__verbose__):
                       print ""
                       print ""
                       print "Change Point: ", cp1, " , ", cp2, " (",maxval, ")"
                   
                   self.__cp1_found__ = cp1
                   self.__cp2_found__ = cp2
                   self.__maxval__ = maxval

                   return 
        
                else:
            
                   cp1stop = time-1
            
                   if self.__cp1_start__ <= 0 or self.__cp1_start__ > time-1:
                       raise Error ("CP1 start invalid value")
                  
                   if self.__cp1_stop__ < 0:
                       cp1stop = time-1
                   else:
                       cp1stop = self.__cp1_stop__
                  
                   if cp1stop <= self.__cp1_start__ or cp1stop > time-1:
                       raise Error ("CP1 stop invalid value")
            
                   cp2stop = time-1
            
                   if self.__cp2_start__ <= 0 or self.__cp2_start__ > time-1:
                       raise Error ("CP2 start invalid value")
                  
                   if self.__cp2_stop__ < 0:
                       cp2stop = time-1
                   else:
                       cp2stop = self.__cp2_stop__
                  
                   if cp2stop <= self.__cp2_start__ or cp2stop > time-1:
                       raise Error ("CP2 stop invalid value")
            
                   if self.__cp2_start__ <= self.__cp1_start__:
                       raise Error ("CP2 start invalid value")
             
                   # I am lazy
                   tot = 0
                   for c_p1 in range(self.__cp1_start__, cp1stop):
                       for c_p2 in range(self.__cp2_start__, cp2stop):
                           tot = tot + 1
        
                   idx = 0
                   self.__allvalues__ = []
                   for c_p1 in range(self.__cp1_start__, cp1stop):
                       for c_p2 in range(self.__cp2_start__, cp2stop):
            
                           start = tempo.time()
                           cstart = tempo.clock()
            
                           try:
                               L1, L2, L3 = self.__compute_cps__ (\
                                       self.__metacommunity__, c_p1, False, c_p2)
                           except Error:
                               raise Error ("Oops! error in the main function") 
                   
                           if (maxval < L1+L2+L3):
                               maxval = L1 + L2 + L3
                               cp1 = c_p1
                               cp2 = c_p2
            
                           end = tempo.time()
                           cend = tempo.clock()
            
                           if (self.__verbose__):
                                if self.__print_iter_info__:
                                    print "%10d of %10d time (%10.5f s CPU time %10.5f s)"%(\
                                            idx+1 , tot, end - start, cend - cstart)
                                else:
                                    basicutils.progress_bar(idx+1, tot)
        
                           if not(setval is None):
                               setval.setValue(100.0*(float(idx+1)/float(tot)))
                               if setval.wasCanceled():
                                   raise Error("Cancelled")
            
                           idx = idx + 1 
             
                           if not (self.__fp__ is None):
                              self.__fp__.write(str(c_p1) + " " + str(c_p2) + " " 
                                   + str(L1+L2+L3) + "\n")
        
                           self.__allvalues__.append((c_p1, c_p2, L1+L2+L3))
        
                   if (self.__verbose__):
                        print ""
                        print ""
                        print "Change Point: ", cp1, " , ", cp2 ," (",maxval, ")"

                   self.__cp1_found__ = cp1
                   self.__cp2_found__ = cp2
                   self.__maxval__ = maxval

                   return 
        
            elif (self.__num_of_cps__ == 3):
                cp1 = 0
                cp2 = 0
                cp3 = 0
            
                if self.__delta_cp__ > 1:
            
                   cp1stop = time-1
            
                   if self.__cp1_start__ <= 0 or self.__cp1_start__ > time-1:
                       raise Error ("CP1 start invalid value")
                  
                   if self.__cp1_stop__ < 0:
                       cp1stop = time-1
                   else:
                       cp1stop = self.__cp1_stop__
                  
                   if cp1stop <= self.__cp1_start__ or cp1stop > time-1:
                       raise Error ("CP1 stop invalid value")
            
                   # I am lazy
                   tot = 0
                   for c_p1 in range(self.__cp1_start__, cp1stop):
                       for c_p2 in range(c_p1 + self.__delta_cp__, time-1):
                           for c_p3 in range(c_p2 + self.__delta_cp__, time-1):
                               tot = tot + 1
            
                   idx = 0
                   self.__allvalues__ = []
                   for c_p1 in range(self.__cp1_start__, cp1stop):
                       for c_p2 in range(c_p1 + self.__delta_cp__, time-1):
                           for c_p3 in range(c_p2 + self.__delta_cp__, time-1):
            
                               start = tempo.time()
                               cstart = tempo.clock()
            
                               try:
                                   L1, L2, L3, L4 = self.__compute_cps__ (\
                                           self.__metacommunity__, \
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
                  
                               if (self.__verbose__):
                                    if self.__print_iter_info__:
                                        print "%10d of %10d time (%10.5f s CPU time %10.5f s)"%(\
                                                idx+1 , tot, end - start, cend - cstart)
                                    else:
                                        basicutils.progress_bar(idx+1, tot)
        
                               if not(setval is None):
                                   setval.setValue(100.0*(float(idx+1)/float(tot)))
                                   if setval.wasCanceled():
                                      raise Error("Cancelled")
                                
                               idx = idx + 1 
                              
                               if not (self.__fp__ is None):
                                  self.__fp__.write(str(c_p1) + " " + str(c_p2) + " " 
                                       + str(c_p3) + " " 
                                       + str(L1+L2+L3+L4) + "\n")
                               self.__allvalues__.append((c_p1, c_p2, c_p3, L1+L2+L3+L4))
        
                   if (self.__verbose__):
                        print ""
                        print ""
                        print "Change Point: ", cp1, " , ", cp2, \
                                " ", cp3, " (",maxval, ")"
        
                   self.__cp1_found__ = cp1
                   self.__cp2_found__ = cp2
                   self.__cp3_found__ = cp3
                   self.__maxval__ = maxval

                   return 
        
                else:
            
                   cp1stop = time-1
            
                   if self.__cp1_start__ <= 0 or self.__cp1_start__ > time-1:
                       raise Error ("CP1 start invalid value")
                  
                   if self.__cp1_stop__ < 0:
                       cp1stop = time-1
                   else:
                       cp1stop = self.__cp1_stop__
                  
                   if cp1stop <= self.__cp1_start__ or cp1stop > time-1:
                       raise Error ("CP1 stop invalid value")
            
                   cp2stop = time-1
            
                   if self.__cp2_start__ <= 0 or self.__cp2_start__ > time-1:
                       raise Error ("CP2 start invalid value")
                  
                   if self.__cp2_stop__ < 0:
                       cp2stop = time-1
                   else:
                       cp2stop = self.__cp2_stop__
                  
                   if cp2stop <= self.__cp2_start__ or cp2stop > time-1:
                       raise Error ("CP2 stop invalid value")
            
                   if self.__cp2_start__ <= self.__cp1_start__:
                       raise Error ("CP2 start invalid value")
            
                   cp3stop = time-1
            
                   if self.__cp3_start__ <= 0 or self.__cp3_start__ > time-1:
                       raise Error ("CP3 start invalid value")
                  
                   if self.__cp3_stop__ < 0:
                       cp3stop = time-1
                   else:
                       cp3stop = self.__cp3_stop__
                  
                   if cp3stop <= self.__cp3_start__ or cp3stop > time-1:
                       raise Error ("CP3 stop invalid value")
            
                   if self.__cp3_start__ <= self.__cp2_start__:
                       raise Error ("CP3 start invalid value")
             
                   tot = 0
                   for c_p1 in range(self.__cp1_start__, cp1stop):
                       for c_p2 in range(self.__cp2_start__, cp2stop):
                           for c_p3 in range(self.__cp3_start__, cp3stop):
                               tot = tot + 1
            
                   idx = 0
                   self.__allvalues__ = []
                   for c_p1 in range(self.__cp1_start__, cp1stop):
                       for c_p2 in range(self.__cp2_start__, cp2stop):
                           for c_p3 in range(self.__cp3_start__, cp3stop):
            
                               start = tempo.time()
                               cstart = tempo.clock()
            
                               try:
                                   L1, L2, L3, L4 = self.__compute_cps__ (\
                                           self.__metacommunity__,  \
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
                  
        
                               if (self.__verbose__):
                                    if self.__print_iter_info__:
                                        print "%10d of %10d time (%10.5f s CPU time %10.5f s)"%(\
                                                idx+1 , tot, end - start, cend - cstart)
                                    else:
                                        basicutils.progress_bar(idx+1, tot)
        
                               if not(setval is None):
                                   setval.setValue(100.0*(float(idx+1)/float(tot)))
                                   if setval.wasCanceled():
                                      raise Error("Cancelled")
        
                               
                               idx = idx + 1 
                               if not (self.__fp__ is None):
                                    self.__fp__.write(str(c_p1) + " " + str(c_p2) + " " \
                                       + str(c_p3) + " " \
                                       + str(L1+L2+L3+L4) + "\n")
                               self.__allvalues__.append((c_p1, c_p2, c_p3, L1+L2+L3+L4))
        
                   if (self.__verbose__):
                        print ""
                        print ""
                        print "Change Point: ", cp1, " , ", cp2 , \
                                " ", cp3, " (",maxval, ")"
                   
                   self.__cp1_found__ = cp1
                   self.__cp2_found__ = cp2
                   self.__cp3_found__ = cp3
                   self.__maxval__ = maxval
 
                   return 
        
        raise Error ("Nothing todo")


###############################################################################
# PRIVATE
###############################################################################

    def __compute_cps__ (self, metacommunity, c_p1, performtest = False, \
            c_p2 = -1, c_p3 = -1):

        countries=metacommunity.shape[0]
        rating=numpy.max(metacommunity)
        time=metacommunity.shape[1]
        
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
                                if (metacommunity[c, t] == (i+1)) and \
                                        (metacommunity[c, t+1] == (j+1)):
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
            den1 = 0.0
            
            L1 = 0.0
            
            for i in range(rating):
                 den1 = 0.0
                 for j in range(rating):
                    for c in range(countries):
                         for t in range(c_p1-1):
                            if (metacommunity[c, t] == (i+1)) and \
                                    (metacommunity[c, t+1] == (j+1)):
                                num1[i, j] = num1[i, j] + 1
                    
                 den1 = numpy.sum(num1[i,:])
            
                 if (den1 > 0.0):
                    vals = numpy.float64(num1[i,:])/numpy.float64(den1)
                    L1 += numpy.sum(num1[i,:]*(numpy.ma.log(vals)).filled(0))
            
            if performtest:

                 if den1 != 0:
                    pr1 = numpy.divide(num1, float(den1))
                 else:
                    numpy.fill_diagonal(pr1, 1)

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
                                  if (metacommunity[c, t] == (i+1)) and \
                                          (metacommunity[c, t+1] == (j+1)):
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
                                      if (metacommunity[c, t] == (i+1)) and \
                                              (metacommunity[c, t+1] == \
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
                                      if (metacommunity[c, t] == (i+1)) and \
                                              (metacommunity[c, t+1] \
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
                                      if (metacommunity[c, t] == (i+1)) and \
                                              (metacommunity[c, t+1] \
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
                                  if (metacommunity[c, t] == (i+1)) and \
                                          (metacommunity[c, t+1] \
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


    def __mkc_prop__ (self, metacommunity, transitions_probability_mtx): 

        mcrows = metacommunity.shape[0]
        mcmaxvalue = numpy.max(metacommunity)
        mccols = metacommunity.shape[1]
        
        cdf = numpy.zeros((mcmaxvalue,mcmaxvalue), dtype='float64')
        
        for i in range (mcmaxvalue):
            cdf[i, 0] = transitions_probability_mtx[i, 0]
        
        for i in range(mcmaxvalue):
            for j in range(1,mcmaxvalue):
                cdf[i, j] = transitions_probability_mtx[i, j] \
                        + cdf[i, j-1]
        
        x = numpy.zeros((mcrows,mccols), dtype='int')
        xi = numpy.random.rand(mcrows,mccols)
        
        for c in range(mcrows):
            x[c, 0] = metacommunity[c, 0]
        
        for c in range(mcrows):
            if xi[c, 0] <= cdf[x[c, 0]-1, 0]:
                x[c, 1] = 1
        
            for k in range(1,mcmaxvalue):
                if (cdf[x[c, 0]-1, k-1] < xi[c, 0]) and \
                        (xi[c, 0] <= cdf[x[c, 0]-1, k] ):
                   x[c, 1] = k + 1
        
            for t in range(2,mccols):
                if xi[c, t-1] <= cdf[x[c, t-1]-1, 0]:
                    x[c, t] = 1
        
                for k in range(1,mcmaxvalue):
                    if (cdf[x[c, t-1]-1, k-1] < xi[c, t-1]) \
                            and (xi[c, t-1] <= cdf[x[c, t-1]-1, k]):
                      x[c, t] = k + 1
        
        return x


