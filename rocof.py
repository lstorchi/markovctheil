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

parser = argparse.ArgumentParser()

parser.add_argument("-m","--rmat-filename", help="Transition probability matrix filename", \
        type=str, required=True, dest="rmatfilename")
parser.add_argument("-M", "--name-of-matrix", help="Name of the probability matrix ", \
        type=str, required=False, default="ms", dest="nameofmatrix")
parser.add_argument("-d", "--dimension", help="Specify dim value ", \
        type=int, required=False, default=3)
parser.add_argument("-a", "--absorb", help="Absord set to true", \
        required=False, default=False, action="store_true")
parser.add_argument("-v", "--verbose", help="increase output verbosity", \
        default=False, action="store_true")

if len(sys.argv) == 1:
    parser.print_help()
    exit(1)

args = parser.parse_args()

verbose = args.verbose
filename1 = args.rmatfilename
namems = args.nameofmatrix

if not (os.path.isfile(filename1)):
    print("File " + filename1 + " does not exist ")
    exit(1)

msd = scipy.io.loadmat(filename1)

if not(namems in msd.keys()):
    print ("Cannot find " + namems + " in " + filename1)
    print (msd.keys())
    exit(1)

ms = msd[namems]

absorb = args.absorb

errmsg = []
dim = args.dimension

val = mainmkvcmp.comp_rocof (ms, dim, absorb, \
        verbose, False, errmsg)

time = ms.shape[1]

if val == None:
    for m in errmsg:
        print (m)
    exit(1)
else:
    for t in range(time):
        sys.stdout.write("%d "%(t+1))
        for i in range(dim):
            sys.stdout.write(" %f "%(val[i,t]))
        sys.stdout.write("\n")

