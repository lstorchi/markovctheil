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
        print(("File " + filename + " does not exist "))
        exit(1)
    
    matf = scipy.io.loadmat(filename)
    
    if not(name_rtmt in list(matf.keys())):
        print("Cannot find " + name_rtmt + " in " + filename)
        print(list(matf.keys()))
        exit(1)
    
    if not(name_spmt in list(matf.keys())):
        print("Cannot find " + name_spmt + " in " + filename)
        print(list(matf.keys()))
        exit(1)
    
    ratings = matf[name_rtmt]
    spread = matf[name_spmt]
    p_rating = matf[name_prmt]

    #kstest.slipshod_kstest (spread, ratings)

    if ratings.shape != spread.shape:
        print("Error  in matrix dimension")
        exit(1)
    
    N = numpy.max(ratings)
    Nnaz = spread.shape[0]
    Dst = max(spread.shape)

    if verbose:
        print("Computing Copula parameters ...")

    G, X, rho = mainmkvcmp.compute_copula_variables (ratings, spread)
    
    # eigvals, m =  numpy.linalg.eig(rho)
    # print numpy.sort(eigvals)
    # copula gaussian Kendall
    # tau = 2.0*numpy.arcsin(rho)/math.pi
    # print tau

    entropy_t = mainmkvcmp.runmcsimulation_copula (ratings, spread, \
            p_rating, G, X, rho, \
            Nnaz, Nsim, d, verbose, None)

    print(numpy.mean(entropy_t[1:, :],1))
    plt.plot(numpy.mean(entropy_t[1:, :],1))
    plt.savefig("mean_entropy.png")
