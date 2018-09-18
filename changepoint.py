import argparse
import sys

import os.path

sys.path.append("./module")
import changemod
import basicutils

parser = argparse.ArgumentParser()

parser.add_argument("-m","--rmat-filename", help="Transition probability matrix filename", \
        type=str, required=True, dest="rmatfilename")
parser.add_argument("-M", "--name-of-matrix", help="Name of the probability matrix ", \
        type=str, required=False, default="ms", dest="nameofmatrix")
parser.add_argument("-c", "--numof-cp", help="Number of change points 1, 2, 3 ", \
        type=int, required=False, default=1, dest="numofcp")

if len(sys.argv) == 1:
    parser.print_help()
    exit(1)

args = parser.parse_args()

if not (os.path.isfile(args.rmatfilename)):
    errmsg.append("File " + args.rmatfilename + " does not exist ")
    exit(1)

msd = scipy.io.loadmat(args.rmatfilename)

if not(namems in msd.keys()):
    print "Cannot find " + args.nameofmatrix + " in " + args.rmatfilename
    print msd.keys()
    exit(1)

ms = msd[namems]

