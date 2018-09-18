import argparse
import numpy
import sys

import scipy.io
import os.path

sys.path.append("./module")
import changemod
import basicutils

parser = argparse.ArgumentParser()

parser.add_argument("-m","--rmat-filename", help="Transition probability matrix filename", \
        type=str, required=True, dest="rmatfilename")
parser.add_argument("-M", "--name-of-matrix", help="Name of the probability matrix (default: ms)", \
        type=str, required=False, default="ms", dest="nameofmatrix")
parser.add_argument("-c", "--numof-cp", help="Number of change points 1, 2, 3 (default: 1)", \
        type=int, required=False, default=1, dest="numofcp")
parser.add_argument("-o", "--output-file", help="Dumps all values (default: change.txt)", \
        type=str, required=False, default="change.txt", dest="outf")
parser.add_argument("--iterations", help="Use iteration number instead of progressbar",
        required=False, default=False, action="store_true")

parser.add_argument("--cp1-start", help="CP 1 start from (default: 1)", \
        type=int, required=False, default=0, dest="cp1start")
parser.add_argument("--cp1-stop", help="CP 1 stop  (default: -1 i.e. will stop at maximum time)", \
        type=int, required=False, default=0, dest="cp1stop")


if len(sys.argv) == 1:
    parser.print_help()
    exit(1)

args = parser.parse_args()

if not (os.path.isfile(args.rmatfilename)):
    errmsg.append("File " + args.rmatfilename + " does not exist ")
    exit(1)

msd = scipy.io.loadmat(args.rmatfilename)

if not(args.nameofmatrix in msd.keys()):
    print "Cannot find " + args.nameofmatrix + " in " + args.rmatfilename
    print msd.keys()
    exit(1)

ms = msd[args.nameofmatrix]

rating = numpy.max(ms)
time = ms.shape[1]

errmsg = ""

fp = open(args.outf, "w")

maxval = -1.0 * float("inf")

if (args.numofcp == 1):

    cp1stop = time-1

    if args.cp1start <= 0 or args.cp1start > time-1:
        print "CP1 start invalid value"
        exit(1)

    if args.cp1stop <= args.cp1start or args.cp1stop > time-1:
        print "CP1 stop invalid value"
        exit(1)

    if args.cp1stop < 0:
        cp1stop = time-1
    else:
        cp1stop = args.cp1stop

    cp = 0
    idx = 0
    for c_p in range(args.cp1start, cp1stop):
        if args.iterations:
            print "%10d of %10d"%(idx+1 , cp1stop-args.cp1start)
        else:
            basicutils.progress_bar(idx+1, cp1stop-args.cp1start)

        L1, L2 = changemod.compute_cps(ms, errmsg, c_p)
                
        if (L1 == None):
            print errmsg
            exit(1)

        if (maxval < L1+L2):
            maxval = L1 + L2
            cp = c_p
    
        fp.write(str(c_p) + " " + str(L1+L2) + "\n")

        idx = idx + 1 

    print ""
    print ""
    print "Change Point: ", cp, " (",maxval, ")"

elif (args.numofcp == 2):
    cp1 = 0
    cp2 = 0


fp.close()
