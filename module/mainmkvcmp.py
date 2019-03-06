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

class markovkernel:

    def __init__ (self):
        """
        Init markovkernel class.
        """

        # input
        self.__metacommunity__ = None 
        self.__attributes__ = None 

        self.__infinite_time__ = False 
        
        self.__use_a_seed__ = False
        self.__seed_value__ = 9001

        self.__verbose__ = False 
        self.__usecopula__ = False 
        self.__dump_files__  = False 

        self.__num_of_mc_iterations__ = 100 
        self.__simulated_time__ = 365 
        self.__step__ = 0.25 

        # output
        self.__entropy__ = None 
        self.__entropy_sigma__ = None 
        self.__transitions_probability_mtx__ = None 

        self.__attributes_pdf_values__ = [] 
        self.__attributes_pdf_bins__ = [] 
        self.__attributes_mean_values__ = [] 
        self.__attributes_sigma_values__ = [] 


    def set_metacommunity (self, inmat):
        if not isinstance(inmat, numpy.ndarray):
            raise TypeError("input must be a numpy array")

        self.__metacommunity__ = inmat


    def get_metacommunity (self):
        return self.__metacommunity__


    def set_attributes (self, inmat):
        if not isinstance(inmat, numpy.ndarray):
            raise TypeError("input must be a numpy array")

        self.__attributes__ = inmat


    def get_attributes (self):
        return self.__attributes__


    def set_infinite_time (self, flagin):
        if not isinstance(flagin, bool):
            raise TypeError("input must be a boolean")

        self.__infinite_time__ = flagin


    def get_infinite_time (self):
        return self.__infinite_time__


    def set_use_a_seed (self, inflag, seedvalue=9001):
        if not isinstance(inflag, bool):
            raise TypeError("input must be a boolean")

        if not isinstance(inflag, int):
            raise TypeError("input must be an integer")

        self.__use_a_seed__ = inflag
        self.__seed_value__ = seedvalue


    def get_use_a_seed (self):
        return (self.__use_a_seed__, self.__seed_value__)


    def set_verbose (self, flagin):
        if not isinstance(flagin, bool):
            raise TypeError("input must be a boolean")

        self.__verbose__ = flagin


    def get_verbose (self):
        return self.__verbose__

 
    def set_usecopula (self, flagin):
        if not isinstance(flagin, bool):
            raise TypeError("input must be a boolean")

        self.__usecompula__ = flagin


    def get_usecopula (self):
        return self.__usecopula__


    def set_dump_files (self, flagin):
        if not isinstance(flagin, bool):
            raise TypeError("input must be a boolean")

        self.__dump_files__ = flagin


    def get_dump_files (self):
        return self.__dump_files__


    def set_num_of_mc_iterations (self, numin):
        if not isinstance(numin, int):
            raise TypeError("input must be an integer")

        self.__num_of_mc_iterations__ = numin

    def get_num_of_mc_iterations (self):
        return self.__num_of_mc_iterations__


    def set_simulated_time (self, numin):
        if not isinstance(numin, int):
            raise TypeError("input must be an integer")

        self.__simulated_time__ = numin


    def get_simulated_time (self):
        return self.__simulated_time__


    def set_step(self, inval):
        if not isinstance(inval, float):
            raise TypeError("input must be a float")

        self.__step__ = inval


    def get_step(self):
        return self.__step__


    def get_entropy (self):
        return self.__entropy__


    def get_entropy_sigma (self):
        return self.__entropy_sigma__


    def get_attributes_pdf_values (self):
        return self.__attributes_pdf_values__ 


    def get_attributes_pdf_bins (self):
        return self.__attributes_pdf_bins__ 


    def get_transitions_probability_mtx (self):
        return self.__transitions_probability_mtx__ 


    def get_attributes_mean_values (self):
        return self.__attributes_mean_values__


    def get_attributes_sigma_values (self):
        return self.__attributes_sigma_values__ 
    
    
    def main_mkc_comp (self, setval=None):

        if self.__use_a_seed__:
            numpy.random.seed(self.__seed_value__)
        
        countries = self.__metacommunity__.shape[0]
        rating = numpy.max(self.__metacommunity__)
        time = self.__metacommunity__.shape[1]
        
        if (rating <= 0) or (rating > 8):
            raise ValueError("metacommunity has invalid values")

        self.__transitions_probability_mtx__ = \
                numpy.zeros((rating,rating), dtype='float64') 
        
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
                        if (self.__metacommunity__[k, t] == (i+1)) \
                                and (self.__metacommunity__[k, t+1] \
                                == (j+1)):
                            nk[i, j, k] = nk[i, j, k] + 1
        
                        num[i, j] = sum(nk[i, j])
        
                    den[i] = sum(num[i])
        
            if self.__verbose__:
              basicutils.progress_bar(k+1, countries)
        
            if setval != None:
              setval.setValue(100.0*(float(k+1)/float(countries)))
              if setval.wasCanceled():
                  #errmsg.append("Cancelled!")
                  return False
        
        if setval != None:
             setval.setValue(0)
             setval.setLabelText("Running...")
        
        for i in range(rating):
            for j in range(rating):
                if den[i] != 0:
                    self.__transitions_probability_mtx__[i, j] = \
                            float(num[i, j])/float(den[i])
                else: 
                    self.__transitions_probability_mtx__[i, j] = 0.0
        
        if self.__infinite_time__: 
            # matrice delle probabilita' diventa stazionaria tempo elevato 
            if self.__verbose__:
              print ""
              print "Solve ..."
        
            ai = numpy.identity(rating, dtype='float64') - \
                    numpy.matrix.transpose(self.__transitions_probability_mtx__)
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
                    self.__transitions_probability_mtx__[i, j] = x[0][j] 
        
        if self.__verbose__:
          print " "
          print "Solve SVD "
        
        npr = self.__transitions_probability_mtx__ - \
                numpy.identity(rating, dtype='float64')
        s, v, d = numpy.linalg.svd(npr)
        
        if self.__verbose__:
            print " "
            print "mean value: ", numpy.mean(v)
        
        for i in range(len(self.__attributes__)):
            for j in range(len(self.__attributes__[0])):
                if math.isnan(self.__attributes__[i, j]):
                   self.__attributes__[i, j] = float('inf')
        
        benchmark = numpy.amin(self.__attributes__, 0)
        
        r = numpy.zeros((countries,time), dtype='float64') 
        
        for i in range(countries):
            for j in range(time):
                r[i, j] = self.__attributes__[i, j] - benchmark[j]
        
        for i in range(len(r)):
            for j in range(len(r[0])):
                if (r[i, j] == float('Inf')):
                   r[i, j] = float('nan')
        
        ist = numpy.zeros((rating,time*countries), dtype='float64')
        nn = numpy.zeros((rating), dtype='int')
        
        if setval != None:
          setval.setValue(50.0)
          if setval.wasCanceled():
              #errmsg.append("Cancelled!")
              return False
        
        for i in range(rating):
            for j in range(countries):
                for k in range(time):
                    if self.__metacommunity__[j, k] == i+1: 
                        nn[i] = nn[i] + 1 
                        ist[i, nn[i]-1] = r[j, k]
        
        y = numpy.zeros((ist.shape[0], nn[0]), dtype='float64')
        for i in range(len(ist)):
            y[i] = ist[i, 0:nn[0]]
        
        tiv = []
        
        fname = ""
        
        if rating > 0:
            if self.__dump_files__:
                fname = "aaa"
        
            a, b, c, d, e = basicutils.extract_ti_mean (y[0, :nn[0]], \
                    self.__step__, 0, self.__num_of_mc_iterations__, \
                    fname)
        
            self.__attributes_pdf_bins__.append(d)
            self.__attributes_pdf_values__.append(a)
            self.__attributes_sigma_values__.append(e)
            self.__attributes_mean_values__.append(b)
            tiv.append(c)
        
        if rating > 1:
            if self.__dump_files__:
                fname = "aa"
        
            a, b, c, d, e = basicutils.extract_ti_mean ( \
                    y[1, :nn[1]], self.__step__, \
                    1, self.__num_of_mc_iterations__, \
                    fname)
        
            self.__attributes_pdf_bins__.append(d)
            self.__attributes_pdf_values__.append(a)
            self.__attributes_sigma_values__.append(e)
            self.__attributes_mean_values__.append(b)
            tiv.append(c)
        
        if rating > 2:
            if self.__dump_files__:
                fname = "a"
        
            a, b, c, d, e = basicutils.extract_ti_mean ( \
                    y[2, :nn[2]], self.__step__, \
                    2, self.__num_of_mc_iterations__, \
                    fname)
        
            self.__attributes_pdf_bins__.append(d)
            self.__attributes_pdf_values__.append(a)
            self.__attributes_sigma_values__.append(e)
            self.__attributes_mean_values__.append(b)
            tiv.append(c)
        
        if rating > 3: 
            if self.__dump_files__:
                fname = "bbb"
        
            a, b, c, d, e = basicutils.extract_ti_mean ( \
                    y[3, :nn[3]], self.__step__, \
                    3, self.__num_of_mc_iterations__, \
                    fname)
        
            self.__attributes_pdf_bins__.append(d)
            self.__attributes_pdf_values__.append(a)
            self.__attributes_sigma_values__.append(e)
            self.__attributes_mean_values__.append(b)
            tiv.append(c)
        
        if rating > 4:
            if self.__dump_files__:
                fname = "bb"
        
            a, b, c, d, e = basicutils.extract_ti_mean ( \
                    y[4, :nn[4]], self.__step__, \
                    4, self.__num_of_mc_iterations__, \
                    fname)
        
            self.__attributes_pdf_bins__.append(d)
            self.__attributes_pdf_values__.append(a)
            self.__attributes_sigma_values__.append(e)
            self.__attributes_mean_values__.append(b)
            tiv.append(c)
        
        if rating > 5:
            if self.__dump_files__:
                fname = "b"
        
            a, b, c, d, e = basicutils.extract_ti_mean ( \
                    y[5, :nn[5]], self.__step__, \
                    5, self.__num_of_mc_iterations__, \
                    fname)
        
            self.__attributes_pdf_bins__.append(d)
            self.__attributes_pdf_values__.append(a)
            self.__attributes_sigma_values__.append(e)
            self.__attributes_mean_values__.append(b)
            tiv.append(c)
        
        if rating > 6:
            if self.__dump_files__:
                fname = "cc"
        
            a, b, c, d, e = basicutils.extract_ti_mean ( \
                    y[6, :nn[6]], self.__step__, \
                    6, self.__num_of_mc_iterations__, \
                    fname)
        
            self.__attributes_pdf_bins__.append(d)
            self.__attributes_pdf_values__.append(a)
            self.__attributes_sigma_values__.append(e)
            self.__attributes_mean_values__.append(b)
            tiv.append(c)
        
        if rating > 7:
            if self.__dump_files__:
                fname = "d"
        
            a, b, c, d, e = basicutils.extract_ti_mean ( \
                    y[7, :nn[7]], self.__step__, \
                    7, self.__num_of_mc_iterations__, \
                    fname)
        
            self.__attributes_pdf_bins__.append(d)
            self.__attributes_pdf_values__.append(a)
            self.__attributes_sigma_values__.append(e)
            self.__attributes_mean_values__.append(b)
            tiv.append(c)
        
        fval = 0.0
        pval = 0.0
        
        if setval != None:
          setval.setValue(75.0)
          if setval.wasCanceled():
              #errmsg.append("Cancelled!")
              return False
        
        args = [] 
        
        for i in range(len(self.__attributes_pdf_values__)):
            args.append(self.__attributes_pdf_values__[i])
        
        fval, pval = scipy.stats.f_oneway (*args)
        
        if self.__verbose__:
          print " "
        
        oufilename = "1wayanova_"+\
                str(self.__num_of_mc_iterations__)+".txt"
        
        if self.__dump_files__:
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
                    T_t[t] += s_t[k, t]*math.log(float(countries) \
                            * s_t[k, t])
        
        oufilename = "entropy_histi_"+\
                str(self.__num_of_mc_iterations__)+".txt"
        
        if self.__dump_files__:
          basicutils.vct_to_file(T_t, oufilename)
        
        if setval != None:
          setval.setValue(100.0)
          if setval.wasCanceled():
              #errmsg.append("Cancelled!")
              return False
        
        
        entropy = None
        ac = None
        bp = None 
        
        if self.__usecopula__:
            G, X, rho = self.__compute_copula_variables__ (\
                    self.__metacommunity__, r)
            
            try:
                entropy = self.__runmcsimulation_copula__ (r, G, \
                    X, rho, countries, \
                    T_t, setval)
            except StopIteration:
                return False

        else:
            try:
               entropy, ac, bp = self.__runmcsimulation__ (\
                    rating, countries, tiv, \
                    setval)
            except StopIteration:
                return False
        
        if self.__verbose__:
          print " "
        
        oufilename = "entropy_"+\
                str(self.__num_of_mc_iterations__)+".txt"

        self.__entropy__ = numpy.zeros(self.__simulated_time__, \
                dtype='float64')
        self.__entropy_sigma__ = numpy.zeros(self.__simulated_time__, \
                dtype='float64')

        for t in range(self.__simulated_time__):
            self.__entropy__[t] = numpy.mean(entropy[t])
            self.__entropy_sigma__[t] = numpy.std(entropy[t])
        
        if self.__dump_files__:
        
          if os.path.exists(oufilename):
              os.remove(oufilename)
        
          outf = open(oufilename, "w")
       
          for t in range(self.__simulated_time__):
              outf.write("%d %f %f \n"%(t+1, self.__entropy__[t], \
                      self.__entropy_sigma__[t]))
         
          outf.close()
        
        if not self.__usecopula__:
        
          acm = numpy.zeros((rating,self.__simulated_time__), \
                  dtype='float64')
          for i in range(acm.shape[0]):
              for j in range(acm.shape[1]):
                  acm[i, j] = numpy.mean(ac[i, j])
          
          oufilename = "acm_"+\
                  str(self.__num_of_mc_iterations__)+".txt"
          
          if self.__dump_files__:
            basicutils.mat_to_file (acm, oufilename)
          
          bpm = numpy.zeros((countries,self.__simulated_time__), \
                  dtype='float64')
          for i in range(bpm.shape[0]):
              for j in range(bpm.shape[1]):
                  bpm[i, j] = numpy.mean(bp[i, j])
          
          oufilename = "bpm_"+\
                  str(self.__num_of_mc_iterations__)+".txt"
         
          if self.__dump_files__:
            basicutils.mat_to_file (bpm, oufilename)
        
        return True


#####################################################################
# PRIVATE
#####################################################################

    def __main_mkc_prop__ (self):

        countries = self.__metacommunity__.shape[0]
        rating = numpy.max(self.__metacommunity__)
        time = self.__metacommunity__.shape[1]
        
        cdf = numpy.zeros((rating,rating), dtype='float64')
        
        for i in range (rating):
            cdf[i, 0] = self.__transitions_probability_mtx__[i, 0]
        
        for i in range(rating):
            for j in range(1,rating):
                cdf[i, j] = self.__transitions_probability_mtx__[i, j] \
                        + cdf[i, j-1]
        
        x = numpy.zeros((countries,time), dtype='int')
        xi = numpy.random.rand(countries,time)
        
        for c in range(countries):
            x[c, 0] = self.__metacommunity__[c, 0]
        
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


    def __compute_copula_variables__ (self, r):

        if self.__metacommunity__.shape != r.shape:
            print "Error  in matrix dimension"
            exit(1)
        
        N = numpy.max(self.__metacommunity__)
        Nnaz = r.shape[0]
        Dst = max(r.shape)
        
        inc_spread = numpy.zeros((Nnaz,Dst-1))
        
        end = r.shape[1]
        for i in range(Nnaz):
            a = r[i,1:end] - r[i,0:end-1]
            b = r[i,0:end-1]
            inc_spread[i,:] = numpy.divide(a, b, \
                    out=numpy.full_like(a, 
                float("Inf")), where=b!=0)
        
        rttmp = self.__metacommunity__[:,1:end]
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
        
            dist_sp = filter(lambda a: a >= mind and a <= maxd, \
                    dist_sp)
        
            x, y = basicutils.ecdf(dist_sp)
            X.append(x)
            G.append(y)
        
        rho = numpy.corrcoef(r) 
        
        return G, X, rho

    
    def __runmcsimulation_copula__ (self, r, G, X, rho, countries, \
        T_t, setval):

        R_in = self.__metacommunity__[:,-1]
        
        spread_synth_tmp = numpy.zeros(countries)
        spread_synth = numpy.zeros(self.__num_of_mc_iterations__,\
                self.__simulated_time__,countries)
        
        entropy = T_t[-1]*numpy.ones((self.__simulated_time__,\
               self.__num_of_mc_iterations__))
        
        if self.__verbose__:
            print "Start MC simulation  ..."
        
        for run in range(self.__num_of_mc_iterations__):
            spread_synth[run,0,:] = r[:,-1].transpose()
            #print spread_synth[run,0,:]
            u = basicutils.gaussian_copula_rnd (rho, \
                    self.__simulated_time__)
        
            for j in range(1,self.__simulated_time__):
                v = numpy.random.uniform(0.0, 1.0, countries)
                pp = numpy.cumsum(\
                        self.__transitions_probability_mtx__[R_in-1,:],1)
                jj = numpy.zeros(countries, dtype=int)
                for k in range(countries):
                    jj[k] = numpy.where(pp[k,:] >= v[k])[0][0]
                    
                    func = interp1d(G[jj[k]][1:], X[jj[k]][1:], \
                            kind='linear')
        
                    xval = u[j,k]
                    xmin = min(G[jj[k]][1:])
                    xmax = max(G[jj[k]][1:])
                    if u[j,k] < xmin:
                        xval = xmin
                    if u[j,k] > xmax:
                        xval = xmax 
        
                    spread_synth_tmp[k] = max(func(xval), -0.9)
        
                R_in = jj
                spread_synth[run,j,:] = numpy.multiply( \
                        numpy.squeeze(spread_synth[run,j-1,:]), \
                        (1+spread_synth_tmp[:].transpose()))
        
                summa = numpy.sum(spread_synth[run,j,:])
                if summa != 0.0:
                    summa = 1.0e-10
        
                P_spread = spread_synth[run,j,:]/numpy.sum(\
                        spread_synth[run,j,:])
        
                P_spread = P_spread.clip(min=1.0e-15)
        
                entropy[j, run] =  numpy.sum(\
                        numpy.multiply(P_spread, \
                        numpy.log(float(countries)*P_spread)))
        
            if self.__verbose__:
                basicutils.progress_bar(run+1, \
                        self.__num_of_mc_iterations__)
            
            if setval != None:
                setval.setValue(100.0*(float(run+1)/ \
                        float(self.__num_of_mc_iterations__)))
                if setval.wasCanceled():
                   #errmsg.append("Cancelled!")
                   raise StopIteration("Cancelled!")

        return entropy
        
    
    def __runmcsimulation__ (self, rating, \
        countries, tiv, setval):

        entropy = numpy.zeros((self.__simulated_time__,\
                self.__num_of_mc_iterations__), dtype='float64')
        
        bp = numpy.zeros((countries,self.__simulated_time__,\
                self.__num_of_mc_iterations__), dtype='float64')
        ac = numpy.zeros((rating,self.__simulated_time__,\
                self.__num_of_mc_iterations__), dtype='float64')
        xm = numpy.zeros((countries,self.__simulated_time__), \
                dtype='float64')
        cdf = numpy.zeros((rating,rating), dtype='float64')
        x = numpy.zeros((countries,self.__simulated_time__), \
                dtype='int')
        r_prev = numpy.zeros((self.__simulated_time__,\
                self.__num_of_mc_iterations__), dtype='float64')
        term = numpy.zeros((self.__simulated_time__,\
                self.__num_of_mc_iterations__), dtype='float64')
        t1 = numpy.zeros((self.__simulated_time__,\
                self.__num_of_mc_iterations__), dtype='float64')
        t2 = numpy.zeros((self.__simulated_time__,\
                self.__num_of_mc_iterations__), dtype='float64')
        
        #print type(G), type(X), type(rho)
       
        for i in range (rating):
            cdf[i, 0] = self.__transitions_probability_mtx__[i, 0]
        
        for i in range(rating):
            for j in range(1,rating):
                cdf[i, j] = self.__transitions_probability_mtx__[i, j] \
                        + cdf[i, j-1]
        
        if setval != None:
             setval.setValue(0)
             setval.setLabelText("Monte Carlo simulation")
        
        for run in range(self.__num_of_mc_iterations__):
        
            tot = numpy.zeros((rating,self.__simulated_time__), \
                    dtype='float64')
            cont = numpy.zeros((rating,self.__simulated_time__), \
                    dtype='int')
            xi = numpy.random.rand(countries,self.__simulated_time__)
            x[:, 0] = self.__metacommunity__[:, -1]
        
            for c in range(countries):
                if xi[c, 0] <= cdf[x[c, 0]-1, 0]:
                    x[c, 1] = 1
        
                for k in range(1,rating):
                    if (cdf[x[c, 0]-1, k-1] < xi[c, 0]) and \
                        (xi[c, 0] <= cdf[x[c, 0]-1, k] ):
                       x[c, 1] = k + 1
        
                for t in range(2,self.__simulated_time__):
                    if xi[c, t-1] <= cdf[x[c, t-1]-1, 0]:
                        x[c, t] = 1
        
                    for k in range(1,rating):
                        if (cdf[x[c, t-1]-1, k-1] < xi[c, t-1]) \
                                and (xi[c, t-1] <= cdf[x[c, t-1]-1, k]):
                          x[c, t] = k + 1
        
            for t in range(self.__simulated_time__):
                for c in range(countries):
                    for i in range(rating):
                        if x[c, t] == i+1:
                            bp[c, t, run] = \
                                    self.__attributes_mean_values__[i]
                            cont[i, t] = cont[i, t] + 1
                            tot[i, t] = cont[i, t] * \
                                    self.__attributes_mean_values__[i]
                    
                r_prev[t, run] = numpy.sum(bp[:, t, run])
        
            for t in range(self.__simulated_time__):
                for i in range(rating):
                     ac[i, t, run] = tot[i, t]/r_prev[t, run]
                     if ac[i, t, run] != 0.0:
                         t1[t, run] += (ac[i, t, run]*tiv[i])
                         t2[t, run] += (ac[i, t, \
                                 run]*math.log(\
                                 float(rating)*ac[i, t, run]))
                         if cont[i, t] != 0:
                            term[t, run] += ac[i, t, run]* \
                                    math.log(float(countries)/ \
                                    (float(rating)*cont[i, t]))
         
                entropy[t, run] = t1[t, run] + t2[t, run] + term[t, run]
        
            if self.__verbose__:
                basicutils.progress_bar(run+1, \
                        self.__num_of_mc_iterations__)
        
            if setval != None:
                setval.setValue(100.0*(float(run+1)/float(\
                        self.__num_of_mc_iterations__)))
                if setval.wasCanceled():
                  #errmsg.append("Cancelled!")
                  raise StopIteration("Cancelled!")
 
         
        return entropy, ac, bp

