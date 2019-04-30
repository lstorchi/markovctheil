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

sys.path.append("./module")
import mainmkvcmp
import basicutils

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-m","--rmat-filename", help="Observed transition matrix filename", \
            type=str, required=True, dest="rmatfilename")
    parser.add_argument("-b", "--imat-filename", help="Rewards matrix filename", \
            type=str, required=True, dest="imatfilename")
    parser.add_argument("-s", "--step", help="Bin width ", \
            type=float, required=False, default=0.25, dest="step")
    parser.add_argument("-t", "--time-prev", help="Simulated period ", \
            type=int, required=False, default=37, dest="tprev")
    parser.add_argument("-n", "--max-run", help="Monte carlo iterations ", \
            type=int, required=True, dest="maxrun")
    parser.add_argument("-M", "--name-of-matrix", help="Name of the observed transition matrix ", \
            type=str, required=False, default="ms", dest="nameofmatrix")
    parser.add_argument("-B", "--name-of-bpmatrix", help="Name of the rewards matrix ", \
            type=str, required=False, default="i_r", dest="nameofbpmatrix")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", \
            default=False, action="store_true")
    parser.add_argument("-i", "--time-inf", help="Simulation using stationary distribution", \
            default=False, action="store_true", dest="timeinf")
    parser.add_argument("-S", "--seed", help="Using a seed for the random generator", \
            default=False, action="store_true", dest="seed")
    parser.add_argument("-c", "--use-copule", help="Using copula", \
            default=False, action="store_true", dest="usecopula")
    
    if len(sys.argv) == 1:
        parser.print_help()
        exit(1)

    args = parser.parse_args()


    namebp = args.nameofbpmatrix
    timeinf = args.timeinf
    verbose = args.verbose
    filename1 = args.rmatfilename
    filename2 = args.imatfilename
    step = args.step
    tprev = args.tprev
    numofrun = args.maxrun
    namems = args.nameofmatrix
    usecopula = args.usecopula
    
    errmsg = []
    
    if not (os.path.isfile(filename1)):
        print("File " + filename1 + " does not exist")
        exit(1)
    
    if not (os.path.isfile(filename2)):
        print("File ", filename2, " does not exist")
        exit(1)

    msd = None
    bpd = None

    if filename1.endswith('.csv'):
        msd = basicutils.csvfile_to_mats(filename1)
        if msd == None:
             print("Error while reading file " + filename1)
             exit(1)
    elif filename1.endswith('.mat'):
        msd = scipy.io.loadmat(filename1)
    else:
        print("Error in file extension")
        exit(1)

    if filename2.endswith('.csv'):
        bpd = basicutils.csvfile_to_mats(filename2)
        if bpd == None:
             print("Error while reading file " + filename2)
             exit(1)
    elif filename2.endswith('.mat'):
        bpd = scipy.io.loadmat(filename2)
    else:
        print("Error in file extension")
        exit(1)

    if not(namems in list(msd.keys())):
        print("Cannot find " + namems + " in " + filename1)
        print(list(msd.keys()))
        exit(1)
    
    if not(namebp in list(bpd.keys())):
        print("Cannot find " + namebp + " in " + filename2)
        print(list(bpd.keys()))
        exit(1)
    
    if msd[namems].shape[0] != bpd[namebp].shape[0]:
        print("wrong dim of the input matrix")
        exit(1)

    ms = msd[namems].astype(numpy.int)
    ir = bpd[namebp].astype(numpy.float)

    try:
        markovrun = mainmkvcmp.markovkernel()

        markovrun.set_metacommunity(ms)
        markovrun.set_attributes(ir)
        markovrun.set_step(step)

        markovrun.set_infinite_time(timeinf)
        markovrun.set_simulated_time(tprev)

        markovrun.set_num_of_mc_iterations(numofrun)
        markovrun.set_use_a_seed(args.seed)
        markovrun.set_usecopula(usecopula)
        markovrun.set_verbose(verbose)
        markovrun.set_dump_files(True)

        if not markovrun.run_computation():
            print("Error in main markov kernel")
            exit(1)

    except TypeError as err:
        print(err)
        exit(1)
