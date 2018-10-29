import argparse
import numpy
import sys

import scipy.io
import os.path

sys.path.append("./module")
import changemod
import basicutils
import mainmkvcmp

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
        type=int, required=False, default=1, dest="cp1start")
parser.add_argument("--cp1-stop", help="CP 1 stop  (default: -1 i.e. will stop at maximum time)", \
        type=int, required=False, default=-1, dest="cp1stop")

parser.add_argument("--cp2-start", help="CP 2 start from (default: 1)", \
        type=int, required=False, default=1, dest="cp2start")
parser.add_argument("--cp2-stop", help="CP 2 stop  (default: -1 i.e. will stop at maximum time)", \
        type=int, required=False, default=-1, dest="cp2stop")

parser.add_argument("--cp3-start", help="CP 3 start from (default: 1)", \
        type=int, required=False, default=1, dest="cp3start")
parser.add_argument("--cp3-stop", help="CP 3 stop  (default: -1 i.e. will stop at maximum time)", \
        type=int, required=False, default=-1, dest="cp3stop")

parser.add_argument("--delta-cp", help="Delta time between CPs (default: 1 no delta)" + 
        "if delta <= 0 will use cp2 and cp3 start and stop values", \
        type=int, required=False, default=1, dest="deltacp")

parser.add_argument("--perform-test", help="Perfom Lambda test for the specified cp1:cp2;cp3;num_of_run "+
        "(default: \"-1;0\" i.e. no test is performed) will use also cp2 and cp3 if --numof-cp is equal to 2 or 3", 
        required=False, type=str, default="-1;0", dest="performtest")


if len(sys.argv) == 1:
    parser.print_help()
    exit(1)

args = parser.parse_args()

if not (os.path.isfile(args.rmatfilename)):
    print "File " + args.rmatfilename + " does not exist "
    exit(1)

msd = scipy.io.loadmat(args.rmatfilename)

if not(args.nameofmatrix in msd.keys()):
    print "Cannot find " + args.nameofmatrix + " in " + args.rmatfilename
    print msd.keys()
    exit(1)

ms = msd[args.nameofmatrix]

fp = open(args.outf, "w")

cp_fortest = -1
num_of_run = 0
cp_fortest_2 = -1
cp_fortest_3 = -1

if len(args.performtest.split(";")) < 1:
    print "Error in --perform-test option"
    exit(1)

cp_fortest = int(args.performtest.split(";")[0])

if cp_fortest >= 0:
    if args.numofcp == 1:
        if len(args.performtest.split(";")) != 2:
            print "Error in --perform-test option"
            exit(1)
    
        num_of_run = int(args.performtest.split(";")[1])
    elif args.numofcp == 2:
        if len(args.performtest.split(";")) != 3:
            print "Error in --perform-test option"
            exit(1)
     
        cp_fortest_2 = int(args.performtest.split(";")[1])
        num_of_run = int(args.performtest.split(";")[2])
    elif args.numofcp == 3:
        if len(args.performtest.split(";")) != 4:
            print "Error in --perform-test option"
            exit(1)
     
        cp_fortest_2 = int(args.performtest.split(";")[1])
        cp_fortest_3 = int(args.performtest.split(";")[2])
        num_of_run = int(args.performtest.split(";")[3])
try:
    changemod.main_compute_cps (ms, num_of_run, cp_fortest, cp_fortest_2, cp_fortest_3, 
        args.numofcp, args.cp1start, args.cp2start, args.cp3start, args.cp1stop, args.cp2stop, 
        args.cp3stop, args.deltacp, args.iterations, fp, True)
except changemod.Error:
    print "Oops! error in the main function" 
    exit(1)
 
fp.close()
