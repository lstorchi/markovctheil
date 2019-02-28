from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import scipy.stats 
import argparse
import scipy.io
import numpy
import math
import sys
import os

import os.path

sys.path.append("./module")

import mainmkvcmp
import basicutils
import kstest

import matplotlib.pyplot as plt

#######################################################################

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-m","--mat-filename", help="Observed transition matrix filename", \
            type=str, required=True, dest="matfilename")
    parser.add_argument("-R", "--name-of-rating-matrix", help="Name of the observed transition matrix ", \
            type=str, required=False, default="ratings", dest="nameofratingmatrix")
    parser.add_argument("-S", "--name-of-spread-matrix", help="Name of the spread matrix ", \
            type=str, required=False, default="spread", dest="nameofspreadmatrix")
    parser.add_argument("-P", "--name-of-prating-matrix", help="Name of the rating matrix ", \
            type=str, required=False, default="p_rating", dest="nameofpratingmatrix")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", \
            default=False, action="store_true")
    parser.add_argument("-d", "--num-of-day", help="Number of days to simulate ", \
            type=int, required=False, default=365*3, dest="time")
    parser.add_argument("-n", "--number-of-MCrun", help="Number of MonteCarlo simulation steps ", \
            type=int, required=False, default=200, dest="numofsim")
 
    if len(sys.argv) == 1:
        parser.print_help()
        exit(1)
    
    args = parser.parse_args()
    
    name_rtmt = args.nameofratingmatrix
    name_spmt = args.nameofspreadmatrix
    name_prmt = args.nameofpratingmatrix
    filename = args.matfilename
    verbose = args.verbose
    d = args.time
    Nsim = args.numofsim
    
    errmsg = []
    
    if not (os.path.isfile(filename)):
        print ("File " + filename + " does not exist ")
        exit(1)
    
    matf = scipy.io.loadmat(filename)
    
    if not(name_rtmt in matf.keys()):
        print "Cannot find " + name_rtmt + " in " + filename
        print matf.keys()
        exit(1)
    
    if not(name_spmt in matf.keys()):
        print "Cannot find " + name_spmt + " in " + filename
        print matf.keys()
        exit(1)
    
    ratings = matf[name_rtmt]
    spread = matf[name_spmt]
    p_rating = matf[name_prmt]

    #kstest.slipshod_kstest (spread, ratings)

    if ratings.shape != spread.shape:
        print "Error  in matrix dimension"
        exit(1)
    
    N = numpy.max(ratings)
    Nnaz = spread.shape[0]
    Dst = max(spread.shape)

    if verbose:
        print "Computing Copula parameters ..."

    G, X, rho = mainmkvcmp.compute_copula_variables (ratings, spread)
    
    # eigvals, m =  numpy.linalg.eig(rho)
    # print numpy.sort(eigvals)
    # copula gaussian Kendall
    # tau = 2.0*numpy.arcsin(rho)/math.pi
    # print tau

    R_in = ratings[:,-1]

    spread_synth_tmp = numpy.zeros(Nnaz)
    spread_synth = numpy.zeros((Nsim,d,Nnaz));

    entropy_t = 0.4*numpy.ones((Nsim,d))

    if verbose:
        print "Start MC simulation  ..."

    for sim in range(Nsim):
        spread_synth[sim,0,:] = spread[:,-1].transpose()
        #print spread_synth[sim,0,:]
        u = basicutils.gaussian_copula_rnd (rho, d)

        for j in range(1,d):
            v = numpy.random.uniform(0.0, 1.0, Nnaz)
            pp = numpy.cumsum(p_rating[R_in-1,:],1)
            jj = numpy.zeros(Nnaz, dtype=int)
            for k in range(Nnaz):
                jj[k] = numpy.where(pp[k,:] >= v[k])[0][0]
                
                func = interp1d(G[jj[k]][1:], X[jj[k]][1:], kind='linear')

                xval = u[j,k]
                xmin = min(G[jj[k]][1:])
                xmax = max(G[jj[k]][1:])
                if u[j,k] < xmin:
                    xval = xmin
                if u[j,k] > xmax:
                    xval = xmax 

                spread_synth_tmp[k] = max(func(xval), -0.9)

            R_in = jj
            spread_synth[sim,j,:] = numpy.multiply( \
                    numpy.squeeze(spread_synth[sim,j-1,:]), \
                    (1+spread_synth_tmp[:].transpose()))

            summa = numpy.sum(spread_synth[sim,j,:])
            if summa != 0.0:
                summa = 1.0e-10

            P_spread = spread_synth[sim,j,:]/numpy.sum(spread_synth[sim,j,:])

            P_spread = P_spread.clip(min=1.0e-15)

            entropy_t[sim,j] =  numpy.sum(numpy.multiply(P_spread, \
                    numpy.log(float(Nnaz)*P_spread)))


    print numpy.mean(entropy_t[:,1:],0)
    plt.plot(numpy.mean(entropy_t[:,1:],0))
    plt.savefig("mean_entropy.png")
