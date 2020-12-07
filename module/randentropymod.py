import numpy.linalg
import numpy.random
import scipy.stats
import scipy.io
import argparse
import numpy
import math
import sys
import os

import time as tempo

import os.path

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

import basicutils

class Error(Exception):
    """Base class for exceptions in this module."""
    pass


###############################################################################
# randentropykernel
###############################################################################

class randentropykernel:

    def __init__ (self):
        """
        Init randentropykernel class.
        """

        # input
        self.__community__ = None 
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

        self.__inter_entropy__ = None 
        self.__intra_entropy__ = None 
 
        self.__attributes_pdf_values__ = [] 
        self.__attributes_pdf_bins__ = [] 
        self.__attributes_mean_values__ = [] 
        self.__attributes_sigma_values__ = [] 


    def set_community (self, inmat):
        """
        Specify the community matrix.
        """

        if not isinstance(inmat, numpy.ndarray):
            raise TypeError("input must be a numpy array")

        self.__community__ = inmat


    def get_community (self):
        return self.__community__


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

        self.__usecopula__ = flagin


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


    def get_inter_entropy (self):
        return self.__inter_entropy__


    def get_intra_entropy (self):
        return self.__intra_entropy__


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
    
    
    def run_computation (self, setval=None):

        if self.__community__.shape != self.__attributes__.shape:
            raise ValueError("Error in community or attributes dimensions")

        if self.__use_a_seed__:
            numpy.random.seed(self.__seed_value__)
        
        mcrows = self.__community__.shape[0]
        mcmaxvalue = numpy.max(self.__community__)
        mccols = self.__community__.shape[1]
        
        if (mcmaxvalue <= 0) or (mcmaxvalue > 8):
            raise ValueError("community has invalid values")

        self.__transitions_probability_mtx__ = \
                numpy.zeros((mcmaxvalue,mcmaxvalue), dtype='float64') 
        
        nk = numpy.zeros((mcmaxvalue,mcmaxvalue,mcrows), dtype='int64')
        num = numpy.zeros((mcmaxvalue,mcmaxvalue), dtype='int64')
        den = numpy.zeros(mcmaxvalue, dtype='int64')
        
        if setval != None:
             setval.setValue(0)
             setval.setLabelText("Historical data analysis")
        
        for k in range(mcrows):
            for t in range(mccols-1):
                for i in range(mcmaxvalue):
                    for j in range(mcmaxvalue):
                        if (self.__community__[k, t] == (i+1)) \
                                and (self.__community__[k, t+1] \
                                == (j+1)):
                            nk[i, j, k] = nk[i, j, k] + 1
        
                        num[i, j] = sum(nk[i, j])
        
                    den[i] = sum(num[i])
        
            if self.__verbose__:
              basicutils.progress_bar(k+1, mcrows)
        
            if setval != None:
              setval.setValue(100.0*(float(k+1)/float(mcrows)))
              if setval.wasCanceled():
                  #errmsg.append("Cancelled!")
                  return False
        
        if setval != None:
             setval.setValue(0)
             setval.setLabelText("Running...")
        
        for i in range(mcmaxvalue):
            for j in range(mcmaxvalue):
                if den[i] != 0:
                    self.__transitions_probability_mtx__[i, j] = \
                            float(num[i, j])/float(den[i])
                else: 
                    self.__transitions_probability_mtx__[i, j] = 0.0
        
        if self.__infinite_time__: 
            # matrice delle probabilita' diventa stazionaria tempo elevato 
            if self.__verbose__:
              print("")
              print("Solve ...")
        
            ai = numpy.identity(mcmaxvalue, dtype='float64') - \
                    numpy.matrix.transpose(self.__transitions_probability_mtx__)
            a = numpy.zeros((mcmaxvalue+1,mcmaxvalue), dtype='float64')
        
            for i in range(mcmaxvalue):
                for j in range(mcmaxvalue):
                    a[i, j] = ai[i, j]
        
            for i in range(mcmaxvalue):
                a[mcmaxvalue, i] = 1.0 
        
            b = numpy.zeros(mcmaxvalue+1, dtype='float64')
            b[mcmaxvalue] = 1.0
            x = numpy.linalg.lstsq(a, b)
            
            for j in range(mcmaxvalue):
                for i in range(mcmaxvalue):
                    self.__transitions_probability_mtx__[i, j] = x[0][j] 
        
        if self.__verbose__:
          print(" ")
          print("Solve SVD ")
        
        npr = self.__transitions_probability_mtx__ - \
                numpy.identity(mcmaxvalue, dtype='float64')
        s, v, d = numpy.linalg.svd(npr)
        
        if self.__verbose__:
            print(" ")
            print("mean value: ", numpy.mean(v))
        
        for i in range(len(self.__attributes__)):
            for j in range(len(self.__attributes__[0])):
                if math.isnan(self.__attributes__[i, j]):
                   self.__attributes__[i, j] = float('inf')
        
        benchmark = numpy.amin(self.__attributes__, 0)
        
        r = numpy.zeros((mcrows,mccols), dtype='float64') 
        
        for i in range(mcrows):
            for j in range(mccols):
                r[i, j] = self.__attributes__[i, j] - benchmark[j]
        
        for i in range(len(r)):
            for j in range(len(r[0])):
                if (r[i, j] == float('Inf')):
                   r[i, j] = float('nan')
        
        ist = numpy.zeros((mcmaxvalue,mccols*mcrows), dtype='float64')
        nn = numpy.zeros((mcmaxvalue), dtype='int')
        
        if setval != None:
          setval.setValue(50.0)
          if setval.wasCanceled():
              #errmsg.append("Cancelled!")
              return False
        
        for i in range(mcmaxvalue):
            for j in range(mcrows):
                for k in range(mccols):
                    if self.__community__[j, k] == i+1: 
                        nn[i] = nn[i] + 1 
                        ist[i, nn[i]-1] = r[j, k]
        
        y = numpy.zeros((ist.shape[0], nn[0]), dtype='float64')
        for i in range(len(ist)):
            y[i] = ist[i, 0:nn[0]]
        
        tiv = []
        
        fname = ""

        fnamemap = {1:"aaa", 2:"aa", 3:"a", 4:"bbb", \
                5:"bb", 6:"b", 7:"c", 8:"d"}

        for imcval in range(0,8):
            if mcmaxvalue > imcval:
                if self.__dump_files__:
                    fname = fnamemap[imcval+1]

                a, b, c, d, e = basicutils.extract_ti_mean (\
                        y[imcval, :nn[imcval]], \
                        self.__step__, imcval, \
                        self.__num_of_mc_iterations__, \
                        fname)
                
                self.__attributes_pdf_bins__.append(d)
                self.__attributes_pdf_values__.append(a)
                self.__attributes_sigma_values__.append(e)
                self.__attributes_mean_values__.append(b)
                tiv.append(c)
 
        
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
          print(" ")
        
        oufilename = "1wayanova_"+\
                str(self.__num_of_mc_iterations__)+".txt"
        
        if self.__dump_files__:
          if os.path.exists(oufilename):
              os.remove(oufilename)
        
          outf = open(oufilename, "w")
        
          outf.write("F-value: %f\n"%fval)
          outf.write("P value: %f\n"%pval)
        
          outf.close()
        
        s_t = numpy.zeros((mcrows,mccols), dtype='float64')
        
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                if math.isnan(r[i, j]):
                    r[i, j] = 0.0
        
        R_t = numpy.sum(r, axis=0)
        t_t = numpy.zeros(mccols, dtype='float64')
        
        for t in range(mccols):
            for k in range(mcrows):
                s_t[k, t] = r[k, t] / R_t[t]
                if s_t[k, t] != 0:
                    t_t[t] += s_t[k, t]*math.log(float(mcrows) \
                            * s_t[k, t])
        
        oufilename = "entropy_histi_"+\
                str(self.__num_of_mc_iterations__)+".txt"
        
        if self.__dump_files__:
          basicutils.vct_to_file(t_t, oufilename)
        
        if setval != None:
          setval.setValue(100.0)
          if setval.wasCanceled():
              #errmsg.append("Cancelled!")
              return False
        
        
        entropy = None
        ac = None
        bp = None 

        if self.__usecopula__:
            if self.__community__.shape != r.shape:
                raise ValueError("Error in matrix dimensions")

            valuevec, binvec, rho = self.__compute_copula_variables__ (r)

            try:
                entropy, self.__intra_entropy__, self.__inter_entropy__ = \
                        self.__runmcsimulation_copula__ (r, valuevec, \
                        binvec, rho, t_t, setval)
            except StopIteration:
                return False

        else:
            try:
               entropy, ac, bp = self.__runmcsimulation__ (\
                    tiv, setval)
            except StopIteration:
                return False
        
        if self.__verbose__:
          print(" ")
        
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

          if self.__usecopula__:
              oufilename = "inter_entropy_"+\
                      str(self.__num_of_mc_iterations__)+".txt"

              if os.path.exists(oufilename):
                  os.remove(oufilename)
            
              outf = open(oufilename, "w")
              for t in range(self.__simulated_time__):
                  outf.write("%d %f \n"%(t+1, self.__inter_entropy__[t])) 
              outf.close()

              oufilename = "intra_entropy_"+\
                      str(self.__num_of_mc_iterations__)+".txt"

              if os.path.exists(oufilename):
                  os.remove(oufilename)
            
              outf = open(oufilename, "w")
              for t in range(self.__simulated_time__):
                  outf.write("%d %f \n"%(t+1, self.__intra_entropy__[t])) 
              outf.close()
 
        
        if not self.__usecopula__:

          acm = numpy.zeros((mcmaxvalue,self.__simulated_time__), \
                  dtype='float64')
          for i in range(acm.shape[0]):
              for j in range(acm.shape[1]):
                  acm[i, j] = numpy.mean(ac[i, j])
          
          oufilename = "acm_"+\
                  str(self.__num_of_mc_iterations__)+".txt"
          
          if self.__dump_files__:
            basicutils.mat_to_file (acm, oufilename)
          
          bpm = numpy.zeros((mcrows,self.__simulated_time__), \
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


    def __compute_copula_variables__ (self, r):

        mcmaxvalue = numpy.max(self.__community__)
        rrows = r.shape[0]
        Dst = max(r.shape)

        inc_spread = numpy.zeros((rrows,Dst-1))
        
        end = r.shape[1]
        for i in range(rrows):
            a = r[i,1:end] - r[i,0:end-1]
            b = r[i,0:end-1]
            inc_spread[i,:] = numpy.divide(a, b, \
                    out=numpy.full_like(a, 
                float("Inf")), where=b!=0)
        
        rttmp = self.__community__[:,1:end]
        totdim = rttmp.shape[0]*rttmp.shape[1]
        
        rttmp = rttmp.reshape(totdim, order='F')
        f_inc_spread = inc_spread.reshape(totdim, order='F')
        
        valuevec = []
        binvec = []
        
        for i in range(mcmaxvalue):
            tmp = numpy.where(rttmp == i+1)[0]
            dist_sp = [f_inc_spread[j] for j in tmp]
            dist_sp = [a for a in dist_sp if a != float("Inf")]
            mind = scipy.stats.mstats.mquantiles(dist_sp, 0.05)
            maxd = scipy.stats.mstats.mquantiles(dist_sp, 0.95)
        
            dist_sp = [a for a in dist_sp if a >= mind and a <= maxd]
        
            x, y = basicutils.ecdf(dist_sp)

            valuevec.append(x)
            binvec.append(y)
        
        rho = numpy.corrcoef(r) 

        return valuevec, binvec, rho

    
    def __runmcsimulation_copula__ (self, r, valuevec, binvec, rho, \
        t_t, setval):

        r_in = self.__community__[:,-1]
        mcrows = self.__community__.shape[0]
        
        spread_synth_tmp = numpy.zeros(mcrows)
        spread_synth = numpy.zeros((self.__num_of_mc_iterations__,\
                self.__simulated_time__, mcrows))

        entropy = t_t[-1]*numpy.ones((self.__simulated_time__,\
               self.__num_of_mc_iterations__), dtype=numpy.float64)

        intra_entr = numpy.zeros((self.__simulated_time__, \
                self.__num_of_mc_iterations__), dtype=numpy.float64)

        inter_entr = numpy.zeros((self.__simulated_time__, \
                self.__num_of_mc_iterations__), dtype=numpy.float64)

        inter_entropy = numpy.zeros(self.__simulated_time__, dtype=numpy.float64)
        intra_entropy = numpy.zeros(self.__simulated_time__, dtype=numpy.float64)

        if self.__verbose__:
            print("Start MC simulation  ...")

        if setval != None:
            setval.setValue(0)
            setval.setLabelText("Monte Carlo simulation")

        maxratval = numpy.max(self.__community__)

        for run in range(self.__num_of_mc_iterations__):
            spread_synth[run,0,:] = r[:,-1].transpose()
            #print spread_synth[run,0,:]
            #print self.__attributes__[:,-1]
            #print self.__community__[:,-1]

            u = basicutils.gaussian_copula_rnd (rho, \
                    self.__simulated_time__)
            
            r_prev = numpy.zeros(self.__simulated_time__,\
                    dtype=numpy.float64)

            r_prev[0] = numpy.sum(spread_synth[run,0,:]) 

            spread_cont_time = numpy.zeros((mcrows, maxratval, \
                    self.__simulated_time__), \
                    dtype=numpy.float64)

            counter_per_ratclass = numpy.zeros((maxratval, \
                    self.__simulated_time__), dtype=numpy.float64)

            sum_spread =  numpy.zeros((maxratval, \
                    self.__simulated_time__), dtype=numpy.float64)
            
            entropy_inter_tmp = numpy.zeros((maxratval, \
                self.__simulated_time__), dtype=numpy.float64)

            if setval != None:
                setval.setValue(100.0*(float(run)/ \
                        float(self.__num_of_mc_iterations__)))
                if setval.wasCanceled():
                   #errmsg.append("Cancelled!")
                   raise StopIteration("Cancelled!")


            for j in range(1,self.__simulated_time__):
                v = numpy.random.uniform(0.0, 1.0, mcrows)
                pp = numpy.cumsum(\
                        self.__transitions_probability_mtx__[r_in-1,:],1)
                jj = numpy.zeros(mcrows, dtype=int)
                for k in range(mcrows):

                    jj[k] = numpy.where(pp[k,:] >= v[k])[0][0]
                    
                    func = interp1d(binvec[jj[k]][1:], valuevec[jj[k]][1:], \
                            kind='linear')
        
                    xval = u[j,k]
                    xmin = min(binvec[jj[k]][1:])
                    xmax = max(binvec[jj[k]][1:])
                    if u[j,k] < xmin:
                        xval = xmin
                    if u[j,k] > xmax:
                        xval = xmax 

                    spread_synth_tmp[k] = max(func(xval), -0.9)
        
                r_in = jj+1 # rating at time j 

                spread_synth[run,j,:] = numpy.multiply( \
                        numpy.squeeze(spread_synth[run,j-1,:]), \
                        (1+spread_synth_tmp[:].transpose()))

                #for idex in range(len(r_in)):
                #    print "%5d %5d %10.5f"%(idex+1, r_in[idex], spread_synth[run,j,idex])
                #print ""

                for ratvalue in range(maxratval):
                    indexes = numpy.where(r_in == ratvalue+1)
                    spread_cont_time[indexes[0],ratvalue, j] = spread_synth[run,j,indexes[0]]
                    sum_spread[ratvalue, j] += spread_synth[run,j,indexes[0]].sum()
                    counter_per_ratclass[ratvalue, j] += len(indexes[0])

                summa = numpy.sum(spread_synth[run,j,:])
                if summa == 0.0:
                    summa = 1.0e-10

                r_prev[j] = summa
        
                P_spread = spread_synth[run,j,:]/summa
        
                P_spread = P_spread.clip(min=1.0e-15)
        
                entropy[j, run] =  numpy.sum(\
                        numpy.multiply(P_spread, \
                        numpy.log(float(mcrows)*P_spread)))

            for k in range(mcrows):
                pinter_spread = numpy.divide(spread_cont_time[k, :,:], \
                        sum_spread[:,:], out=numpy.zeros_like(spread_cont_time[k, :,:]), \
                        where=sum_spread[:,:]!=0)
                logm = numpy.ma.log(counter_per_ratclass * pinter_spread)
                entropy_inter_tmp += pinter_spread * logm.filled(0.0)

            acmtx = numpy.divide(sum_spread[:, :], r_prev)

            t1 = acmtx[:,:] * entropy_inter_tmp[:,:]
            mat2 = numpy.ma.log(float(maxratval) * acmtx[:,:])
            t2 = acmtx[:,:] * mat2.filled(0.0)
            dividemat = numpy.divide(float(mcrows), \
                    (float(maxratval)*counter_per_ratclass[:,:]), \
                    out=numpy.zeros_like(counter_per_ratclass[:,:]), \
                    where=counter_per_ratclass[:,:]!=0)
            mat3 = numpy.ma.log(dividemat)
            t3 = acmtx[:,:] * mat3.filled(0.0)

            intra_entr[:,run] = numpy.sum(t1, axis=0)
            inter_entr[:,run] = numpy.sum(t2, axis=0) + numpy.sum(t3, axis=0)

            if self.__verbose__:
                basicutils.progress_bar(run+1, \
                        self.__num_of_mc_iterations__)
            
            if setval != None:
                setval.setValue(100.0*(float(run+1)/ \
                        float(self.__num_of_mc_iterations__)))
                if setval.wasCanceled():
                   #errmsg.append("Cancelled!")
                   raise StopIteration("Cancelled!")

        # add inter and intra entropy computation for the historical values 
        intra_entropy = numpy.mean(intra_entr, axis=1)
        inter_entropy = numpy.mean(inter_entr, axis=1)

        return entropy, intra_entropy, inter_entropy
        
    
    def __runmcsimulation__ (self, tiv, setval):

        entropy = numpy.zeros((self.__simulated_time__,\
                self.__num_of_mc_iterations__), dtype='float64')
        mcmaxvalue = numpy.max(self.__community__)
        mcrows = self.__community__.shape[0]
        
        bp = numpy.zeros((mcrows,self.__simulated_time__,\
                self.__num_of_mc_iterations__), dtype='float64')
        ac = numpy.zeros((mcmaxvalue,self.__simulated_time__,\
                self.__num_of_mc_iterations__), dtype='float64')
        xm = numpy.zeros((mcrows,self.__simulated_time__), \
                dtype='float64')
        cdf = numpy.zeros((mcmaxvalue,mcmaxvalue), dtype='float64')
        x = numpy.zeros((mcrows,self.__simulated_time__), \
                dtype='int')
        r_prev = numpy.zeros((self.__simulated_time__,\
                self.__num_of_mc_iterations__), dtype='float64')
        term = numpy.zeros((self.__simulated_time__,\
                self.__num_of_mc_iterations__), dtype='float64')
        t1 = numpy.zeros((self.__simulated_time__,\
                self.__num_of_mc_iterations__), dtype='float64')
        t2 = numpy.zeros((self.__simulated_time__,\
                self.__num_of_mc_iterations__), dtype='float64')
        
        for i in range (mcmaxvalue):
            cdf[i, 0] = self.__transitions_probability_mtx__[i, 0]
        
        for i in range(mcmaxvalue):
            for j in range(1,mcmaxvalue):
                cdf[i, j] = self.__transitions_probability_mtx__[i, j] \
                        + cdf[i, j-1]
        
        if setval != None:
             setval.setValue(0)
             setval.setLabelText("Monte Carlo simulation")
        
        for run in range(self.__num_of_mc_iterations__):
        
            tot = numpy.zeros((mcmaxvalue,self.__simulated_time__), \
                    dtype='float64')
            cont = numpy.zeros((mcmaxvalue,self.__simulated_time__), \
                    dtype='int')
            xi = numpy.random.rand(mcrows,self.__simulated_time__)
            x[:, 0] = self.__community__[:, -1]
        
            for c in range(mcrows):
                if xi[c, 0] <= cdf[x[c, 0]-1, 0]:
                    x[c, 1] = 1
        
                for k in range(1,mcmaxvalue):
                    if (cdf[x[c, 0]-1, k-1] < xi[c, 0]) and \
                        (xi[c, 0] <= cdf[x[c, 0]-1, k] ):
                       x[c, 1] = k + 1
        
                for t in range(2,self.__simulated_time__):
                    if xi[c, t-1] <= cdf[x[c, t-1]-1, 0]:
                        x[c, t] = 1
        
                    for k in range(1,mcmaxvalue):
                        if (cdf[x[c, t-1]-1, k-1] < xi[c, t-1]) \
                                and (xi[c, t-1] <= cdf[x[c, t-1]-1, k]):
                          x[c, t] = k + 1
        
            for t in range(self.__simulated_time__):
                for c in range(mcrows):
                    for i in range(mcmaxvalue):
                        if x[c, t] == i+1:
                            bp[c, t, run] = \
                                    self.__attributes_mean_values__[i]
                            cont[i, t] = cont[i, t] + 1
                            tot[i, t] = cont[i, t] * \
                                    self.__attributes_mean_values__[i]
                    
                r_prev[t, run] = numpy.sum(bp[:, t, run])
        
            for t in range(self.__simulated_time__):
                for i in range(mcmaxvalue):
                     ac[i, t, run] = tot[i, t]/r_prev[t, run]
                     if ac[i, t, run] != 0.0:
                         t1[t, run] += (ac[i, t, run]*tiv[i])
                         t2[t, run] += (ac[i, t, \
                                 run]*math.log(\
                                 float(mcmaxvalue)*ac[i, t, run]))
                         if cont[i, t] != 0:
                            term[t, run] += ac[i, t, run]* \
                                    math.log(float(mcrows)/ \
                                    (float(mcmaxvalue)*cont[i, t]))
         
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

###############################################################################
# changepoint
###############################################################################

class changepoint:

    def __init__(self):

        self.__community__ = None 
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

    def set_community(self, inmat):
        if not isinstance(inmat, numpy.ndarray):
            raise TypeError("set_community: input must be a numpy array")

        self.__community__ = inmat


    def get_community(self):

        return self.__community__

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
        
        rating = numpy.max(self.__community__)
        time = self.__community__.shape[1]
        
        if self.__cp_fortest_1__ >= 0 and self.__num_of_bootstrap_iter__ >= 0:
        
            if (self.__verbose__):
                print("Startint CP test...")
        
            L = 0.0
            L1 = 0.0
            L2 = 0.0
            L3 = 0.0
            L4 = 0.0
        
            pr1 = 0.0
            self.__lambdastart__ = 0.0
        
            if self.__num_of_cps__ == 1:
                L, L1, L2, pr1, pr2 = self.__compute_cps__ (\
                        self.__community__, self.__cp_fortest_1__, True)
                self.__lambdastart__ = 2.0*((L1+L2)-L)
            elif self.__num_of_cps__ == 2:
                L, L1, L2, L3, pr1, pr2, pr3 = \
                self.__compute_cps__ (self.__community__, \
                self.__cp_fortest_1__, True, self.__cp_fortest_2__)
                self.__lambdastart__ = 2.0*((L1+L2+L3)-L)
                if (self.__verbose__):
                    print(L, L1, L2, L3)
            elif self.__num_of_cps__ == 3:
                L, L1, L2, L3, L4, pr1, pr2, pr3, pr4 = \
                        self.__compute_cps__ (self.__community__, \
                        self.__cp_fortest_1__, True, \
                        elf.__cp_fortest_2__, self.__cp_fortest_3__)
                self.__lambdastart__ = 2.0*((L1+L2+L3+L4)-L)
                if (self.__verbose__):
                    print(L, L1, L2, L3, L4)
        
            lambdas = []
        
            if (self.__verbose__):
                print("Starting iterations...")
        
            for i in range(self.__num_of_bootstrap_iter__):
        
                start = tempo.time()
                cstart = tempo.process_time()
         
                x = self.__mkc_prop__ (self.__community__, pr1)
        
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
                cend = tempo.process_time()
            
                if (self.__verbose__):
                    print("%10d of %10d time (%10.5f s CPU time %10.5f s)"%(\
                            i+1 , self.__num_of_bootstrap_iter__, 
                         end - start, cend - cstart))
         
        
            idx95 = int(self.__num_of_bootstrap_iter__*0.95+0.5)
        
            if idx95 >= self.__num_of_bootstrap_iter__:
                idx95 = self.__num_of_bootstrap_iter__ - 1
        
            self.__lambda95__ = lambdas[idx95]
        
            minrat = numpy.min(self.__community__)
            maxrat = numpy.max(self.__community__)
        
            #ndof = (maxrat - minrat + 1) * (maxrat - minrat)
            #ndof = max (positive1, positive2)
        
            #chi2 = scipy.stats.chi2.isf(0.05, ndof)
            #pvalue = 1.0 - scipy.stats.chi2.cdf(self.__lambda95__, ndof)
        
            self.__pvalue__ = \
                    (1.0 / numpy.float64(self.__num_of_bootstrap_iter__ + 1)) * \
                    (1.0 + numpy.float64(sum(i >= self.__lambdastart__ for i in lambdas)))
        
            if (self.__verbose__):
                #print "Ndof        : ", ndof
                print("Lambda(95%) : ", self.__lambda95__)
                print("Lambda      : ", self.__lambdastart__)
                print("P-Value     : ", self.__pvalue__)
        
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
                    cstart = tempo.process_time()
        
            
                    try:
                        L1, L2 = self.__compute_cps__ (self.__community__, c_p)
                    except Error:
                        raise Error ("Oops! error in the main function") 
            
                    if (maxval < L1+L2):
                        maxval = L1 + L2
                        cp = c_p
                
                    if not (self.__fp__ is None):
                        self.__fp__.write(str(c_p) + " " + str(L1+L2) + "\n")
                    self.__allvalues__.append((c_p, L1+L2))
            
                    end = tempo.time()
                    cend = tempo.process_time()
            
                    if (self.__verbose__):
                        if self.__print_iter_info__:
                            print("%10d of %10d time (%10.5f s CPU time %10.5f s)"%(\
                                    idx+1 , cp1stop-self.__cp1_start__, \
                                    end - start, cend - cstart))
                        else:
                            basicutils.progress_bar(idx+1, cp1stop-self.__cp1_start__)
        
                    if not(setval is None):
                        setval.setValue(100.0*(float(idx+1)/float(cp1stop-self.__cp1_start__)))
                        if setval.wasCanceled():
                            raise Error("Cancelled")
        
                    idx = idx + 1 
            
                if (self.__verbose__):
                    print("")
                    print("")
                    print("Change Point: ", cp, " (",maxval, ")")
                
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
                           cstart = tempo.process_time()
            
                           try:
                               L1, L2, L3 = self.__compute_cps__ (\
                                       self.__community__, c_p1, False, c_p2)
                           except Error:
                               raise Error ("Oops! error in the main function")
             
                           if (maxval < L1+L2+L3):
                               maxval = L1 + L2 + L3
                               cp1 = c_p1
                               cp2 = c_p2
            
                           end = tempo.time()
                           cend = tempo.process_time()
            
                           if (self.__verbose__):
                                if self.__print_iter_info__:
                                    print("%10d of %10d time (%10.5f s CPU time %10.5f s)"%(\
                                            idx+1 , tot, end - start, cend - cstart))
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
                       print("")
                       print("")
                       print("Change Point: ", cp1, " , ", cp2, " (",maxval, ")")
                   
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
                           cstart = tempo.process_time()
            
                           try:
                               L1, L2, L3 = self.__compute_cps__ (\
                                       self.__community__, c_p1, False, c_p2)
                           except Error:
                               raise Error ("Oops! error in the main function") 
                   
                           if (maxval < L1+L2+L3):
                               maxval = L1 + L2 + L3
                               cp1 = c_p1
                               cp2 = c_p2
            
                           end = tempo.time()
                           cend = tempo.process_time()
            
                           if (self.__verbose__):
                                if self.__print_iter_info__:
                                    print("%10d of %10d time (%10.5f s CPU time %10.5f s)"%(\
                                            idx+1 , tot, end - start, cend - cstart))
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
                        print("")
                        print("")
                        print("Change Point: ", cp1, " , ", cp2 ," (",maxval, ")")

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
                               cstart = tempo.process_time()
            
                               try:
                                   L1, L2, L3, L4 = self.__compute_cps__ (\
                                           self.__community__, \
                                           c_p1, False, c_p2, c_p3)
                               except Error:
                                   raise Error ("Oops! error in the main function") 
                               
                               if (maxval < L1+L2+L3+L4):
                                   maxval = L1 + L2 + L3 + L4
                                   cp1 = c_p1
                                   cp2 = c_p2
                                   cp3 = c_p3
            
                               end = tempo.time()
                               cend = tempo.process_time()
                  
                               if (self.__verbose__):
                                    if self.__print_iter_info__:
                                        print("%10d of %10d time (%10.5f s CPU time %10.5f s)"%(\
                                                idx+1 , tot, end - start, cend - cstart))
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
                        print("")
                        print("")
                        print("Change Point: ", cp1, " , ", cp2, \
                                " ", cp3, " (",maxval, ")")
        
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
                               cstart = tempo.process_time()
            
                               try:
                                   L1, L2, L3, L4 = self.__compute_cps__ (\
                                           self.__community__,  \
                                           c_p1, False, c_p2, c_p3)
                               except Error:
                                   raise Error ("Oops! error in the main function") 
             
                               if (maxval < L1+L2+L3+L4):
                                   maxval = L1 + L2 + L3 + L4
                                   cp1 = c_p1
                                   cp2 = c_p2
                                   cp3 = c_p3
            
                               end = tempo.time()
                               cend = tempo.process_time()
                  
        
                               if (self.__verbose__):
                                    if self.__print_iter_info__:
                                        print("%10d of %10d time (%10.5f s CPU time %10.5f s)"%(\
                                                idx+1 , tot, end - start, cend - cstart))
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
                        print("")
                        print("")
                        print("Change Point: ", cp1, " , ", cp2 , \
                                " ", cp3, " (",maxval, ")")
                   
                   self.__cp1_found__ = cp1
                   self.__cp2_found__ = cp2
                   self.__cp3_found__ = cp3
                   self.__maxval__ = maxval
 
                   return 
        
        raise Error ("Nothing todo")


###############################################################################
# PRIVATE
###############################################################################

    def __compute_cps__ (self, community, c_p1, performtest = False, \
            c_p2 = -1, c_p3 = -1):

        countries=community.shape[0]
        rating=numpy.max(community)
        time=community.shape[1]
        
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
                                if (community[c, t] == (i+1)) and \
                                        (community[c, t+1] == (j+1)):
                                    nk[i, j, c] = nk[i, j, c] + 1
                                
                        num[i, j] = numpy.sum(nk[i, j, :])
                    
                    den[i] = numpy.sum(num[i, :])
                
                    if (den[i] > 0.0):
                        vals = numpy.divide(numpy.float64(num[i,:]), \
                                numpy.float64(den[i]))
                        L += numpy.sum(num[i,:] * (numpy.ma.log(vals)).filled(0))

                for i in range(rating):
                    if den[i] != 0:
                        pr[i, :] = numpy.divide(numpy.float64(num[i, :]), \
                                numpy.float64(den[i])) 
                    else:
                        pr[i, :] = 0
                        pr[i, i] = 1

            num1 = numpy.zeros((rating,rating), dtype='int64')
            den1 = 0.0
            
            L1 = 0.0
            
            for i in range(rating):
                 den1 = 0.0
                 for j in range(rating):
                    for c in range(countries):
                         for t in range(c_p1-1):
                            if (community[c, t] == (i+1)) and \
                                    (community[c, t+1] == (j+1)):
                                num1[i, j] = num1[i, j] + 1
                    
                 den1 = numpy.sum(num1[i,:])
            
                 if den1 != 0 :
                   vals = numpy.divide(numpy.float64(num1[i,:]), 
                          numpy.float64(den1))
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
        
                pr3 = None

                if performtest:
                    pr3 = numpy.zeros((rating,rating),dtype='float64')
               
                for i in range(rating):
                     for j in range(rating):
                         for c in range(countries):
                              for t in range(c_p1,c_p2-1) :
                                  if (community[c, t] == (i+1)) and \
                                          (community[c, t+1] == (j+1)):
                                      num2[i, j] = num2[i, j] + 1
                         
                     den2[i] = numpy.sum(num2[i, :])
                
                     if den2[i] != 0 :
                       vals = numpy.divide(numpy.float64(num2[i,:]), \
                               numpy.float64(den2[i]))
                       L2 += numpy.sum(num2[i,:] * (numpy.ma.log(vals)).filled(0))

                if performtest:
        
                     for i in range(rating):
                         if den2[i] != 0:
                             pr2[i, :] = numpy.divide(numpy.float64(num2[i,:]) ,
                                     numpy.float64(den2[i]))
                         else:
                             pr2[i,:] = 0
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
                                      if (community[c, t] == (i+1)) and \
                                              (community[c, t+1] == \
                                              (j+1)):
                                          num3[i, j] = num3[i, j] + 1
                             
                         den3[i] = numpy.sum(num3[i, j])
                    
                         if (den3[i] > 0.0):
                             vals = numpy.divide(numpy.float64(num3[i,j]), 
                                    numpy.float64(den3[i]))
                             L3 += numpy.sum(num3[i,:] * (numpy.ma.log(vals)).filled(0))
        
                    if performtest:
        
                        for i in range(rating):
                            if den3[i] != 0:
                                pr3[i, :] = numpy.divide(numpy.float64(num3[i, :]),
                                      float(den3[i]))
                            else:
                                pr3[i, :] = 0
                                pr3[i, i] = 1

                    
                    num4 = numpy.zeros((rating,rating), dtype='int64')
                    den4 = numpy.zeros(rating, dtype='int64')
                     
                    L4 = 0.0 
                    
                    for i in range(rating):
                         for j in range(rating):
                             for c in range(countries):
                                  for t in range(c_p3,time-1) :
                                      if (community[c, t] == (i+1)) and \
                                              (community[c, t+1] \
                                              == (j+1)):
                                          num4[i, j] = num4[i, j] + 1
                             
                         den4[i] = numpy.sum(num4[i, :])

                         if (den4[i] > 0.0):
                             vals = numpy.divide(numpy.float64(num4[i,:]), 
                                     numpy.float64(den4[i]))
                             L4 += numpy.sum(num4[i,:] * (numpy.ma.log(vals)).filled(0))
        
                    if performtest:
        
                         for i in range(rating):
                             if den4[i] != 0:
                                 pr4[i, :] = numpy.divide(numpy.float64(num4[i,:]),
                                         numpy.float64(den4[i]))
                             else:
                                 pr4[i, :] = 0
                                 pr4[i, i] = 1
        
                         return L, L1, L2, L3, L4, pr1, pr2, pr3, pr4
                    
                    return L1, L2, L3, L4
        
                else:
                    for i in range(rating):
                         for j in range(rating):
                             for c in range(countries):
                                  for t in range(c_p2,time-1) :
                                      if (community[c, t] == (i+1)) and \
                                              (community[c, t+1] \
                                              == (j+1)):
                                          num3[i, j] = num3[i, j] + 1
                             
                         den3[i] = numpy.sum(num3[i, :])
                    
                         if (den3[i] > 0.0):
                             vals = numpy.divide(numpy.float64(num3[i,:]), 
                                     numpy.float64(den3[i]))
                             L3 += numpy.sum(num3[i,:] * (numpy.ma.log(vals)).filled(0))
 
                    if performtest:
        
                        for i in range(rating):
                             if den3[i] != 0:
                                 pr3[i, :] = numpy.divide(numpy.float64(num3[i,:]),
                                         numpy.float64(den3[i]))
                             else:
                                 pr3[i, :] = 0
                                 pr3[i, i] = 1
 
                        return L, L1, L2, L3, pr1, pr2, pr3
        
                    return L1, L2, L3 
        
            else:
        
                for i in range(rating):
                     for j in range(rating):
                         for c in range(countries):
                              for t in range(c_p1,time-1) :
                                  if (community[c, t] == (i+1)) and \
                                          (community[c, t+1] \
                                          == (j+1)):
                                      num2[i, j] = num2[i, j] + 1
                         
                     den2[i] = numpy.sum(num2[i, :])
                
                     if den2[i] != 0 :
                       vals = numpy.divide(numpy.float64(num2[i,:]), \
                               numpy.float64(den2[i]))
                       L2 += numpy.sum(num2[i,:] * (numpy.ma.log(vals)).filled(0))

                if performtest:
        
                     for i in range(rating):
                         if den2[i] != 0:
                             pr2[i, :] = numpy.divide(numpy.float64(num2[i,:]) ,
                                     numpy.float64(den2[i]))
                         else:
                             pr2[i,:] = 0
                             pr2[i,i] = 1

                     return L, L1, L2, pr1, pr2
 
                
                return L1, L2
        
        raise Error ("at least cp1 should be > 0")


    def __mkc_prop__ (self, community, transitions_probability_mtx): 

        mcrows = community.shape[0]
        mcmaxvalue = numpy.max(community)
        mccols = community.shape[1]
        
        cdf = numpy.zeros((mcmaxvalue,mcmaxvalue), dtype='float64')
        
        cdf[:, 0] = transitions_probability_mtx[:, 0]
        
        for j in range(1,mcmaxvalue):
            cdf[:, j] = transitions_probability_mtx[:, j] \
                + cdf[:, j-1]
        
        x = numpy.zeros((mcrows,mccols), dtype='int')
        xi = numpy.random.rand(mcrows,mccols)
        
        for c in range(mcrows):
            x[c, 0] = community[c, 0]
        
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


