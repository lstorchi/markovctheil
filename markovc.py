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

import mainmkvcmp

sys.path.append("./module")
import basicutils

parser = argparse.ArgumentParser()

parser.add_argument("-m","--msmat-filename", help="MS mat filename", \
        type=str, required=True, dest="msmatfilename")
parser.add_argument("-b", "--bpmat-filename", help="BP mat filename", \
        type=str, required=True, dest="bpmatfilename")
parser.add_argument("-s", "--step", help="Step ", \
        type=float, required=False, default=0.25, dest="step")
parser.add_argument("-t", "--time-prev", help="Time prev ", \
        type=int, required=False, default=37, dest="tprev")
parser.add_argument("-n", "--max-run", help="Num. of run required ", \
        type=int, required=True, dest="maxrun")
parser.add_argument("-M", "--name-of-matrix", help="Name of MS matrix ", \
        type=str, required=False, default="ms", dest="nameofmatrix")
parser.add_argument("-B", "--name-of-bpmatrix", help="Name of BP matrix ", \
        type=str, required=False, default="i_r", dest="nameofbpmatrix")
parser.add_argument("-v", "--verbose", help="increase output verbosity", \
        default=False, action="store_true")
parser.add_argument("-i", "--time-inf", help="Simulate infinie time", \
        default=False, action="store_true", dest="timeinf")
parser.add_argument("-S", "--seed", help="using a seed for the random generator", \
        default=False, action="store_true", dest="seed")

if len(sys.argv) == 1:
    parser.print_help()
    exit(1)

args = parser.parse_args()

namebp = args.nameofbpmatrix
timeinf = args.timeinf
verbose = args.verbose
filename1 = args.msmatfilename
filename2 = args.bpmatfilename
step = args.step
tprev = args.tprev
numofrun = args.maxrun
namems = args.nameofmatrix

errmsg = []

if not mainmkvcmp.main_mkc_comp (filename1, namems, filename2, namebp, \
        timeinf, step, tprev, numofrun, verbose, args.seed, errmsg):
    for m in errmsg:
        print m
    exit(1)

