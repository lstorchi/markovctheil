import argparse
import numpy
import time as tempo
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

rating = numpy.max(ms)
time = ms.shape[1]

fp = open(args.outf, "w")

cp_fortest = -1
num_of_run = 0
cp_fortest_2 = -1
cp_fortest_3 = -1

if args.numofcp == 1:
    if len(args.performtest.split(";")) != 2:
        print "Error in --perform-test option"
        exit(1)

    cp_fortest = int(args.performtest.split(";")[0])
    num_of_run = int(args.performtest.split(";")[1])
elif args.numofcp == 2:
    if len(args.performtest.split(";")) != 3:
        print "Error in --perform-test option"
        exit(1)
 
    cp_fortest = int(args.performtest.split(";")[0])
    cp_fortest_2 = int(args.performtest.split(";")[1])
    num_of_run = int(args.performtest.split(";")[2])
elif args.numofcp == 3:
    if len(args.performtest.split(";")) != 4:
        print "Error in --perform-test option"
        exit(1)
 
    cp_fortest = int(args.performtest.split(";")[0])
    cp_fortest_2 = int(args.performtest.split(";")[1])
    cp_fortest_3 = int(args.performtest.split(";")[2])
    num_of_run = int(args.performtest.split(";")[3])

if cp_fortest >= 0 and num_of_run >= 0:

    print "Startint CP test..."

    L = 0.0
    L1 = 0.0
    L2 = 0.0
    L3 = 0.0
    L4 = 0.0

    pr1 = 0.0
    lambdastart = 0.0

    if args.numofcp == 1:
        L, L1, L2, pr1, pr2 = changemod.compute_cps(ms, cp_fortest, True)
        lambdastart = 2.0*((L1+L2)-L)
    elif args.numofcp == 2:
        L, L1, L2, L3, pr1, pr2, pr3 = \
        changemod.compute_cps(ms, cp_fortest, True, cp_fortest_2)
        lambdastart = 2.0*((L1+L2+L3)-L)
    elif args.numofcp == 3:
        L, L1, L2, L3, L4, pr1, pr2, pr3, pr4 = \
                changemod.compute_cps(ms, cp_fortest, True, \
                cp_fortest_2, cp_fortest_3)
        lambdastart = 2.0*((L1+L2+L3+L4)-L)

    lambdas = []

    print "Starting iterations..."

    for i in range(num_of_run):

        start = tempo.time()
        cstart = tempo.clock()
 
        x = mainmkvcmp.main_mkc_prop (ms, pr1)

        lambdav = 0.0

        L, L1, L2, pr1_o, pr2_o = changemod.compute_cps(x, cp_fortest, True)

        if args.numofcp == 1:
            L, L1, L2, pr1_o, pr2_o = changemod.compute_cps(x, cp_fortest, True)
            lambdav = 2.0*((L1+L2)-L)
        elif args.numofcp == 2:
            L, L1, L2, L3, pr1_o, pr2_o, pr3_o = \
            changemod.compute_cps(x, cp_fortest, True, cp_fortest_2)
            lambdav = 2.0*((L1+L2+L3)-L)
        elif args.numofcp == 3:
            L, L1, L2, L3, L4, pr1_o, pr2_o, pr3_o, pr4_o = \
                    changemod.compute_cps(x, cp_fortest, True, \
                    cp_fortest_2, cp_fortest_3)
            lambdav = 2.0*((L1+L2+L3+L4)-L)

        lambdas.append(lambdav) 

        fp.write(str(i+1) + " " + str(lambdav) + "\n") 

        end = tempo.time()
        cend = tempo.clock()
    
        print "%10d of %10d time (%10.5f s CPU time %10.5f s)"%(i+1 , num_of_run, 
                 end - start, cend - cstart)
 

    idx95 = int(num_of_run*0.95+0.5)

    if idx95 >= num_of_run:
        idx95 = num_of_run - 1

    lamda95 = lambdas[idx95]

    minrat = numpy.min(ms)
    maxrat = numpy.max(ms)

    #ndof = (maxrat - minrat + 1) * (maxrat - minrat)
    #ndof = max (positive1, positive2)

    #chi2 = scipy.stats.chi2.isf(0.05, ndof)
    #pvalue = 1.0 - scipy.stats.chi2.cdf(lamda95, ndof)

    pvalue = (1.0 / numpy.float64(num_of_run + 1)) * \
            (1.0 + numpy.float64(sum(i >= lambdastart for i in lambdas)))

    #print "Ndof        : ", ndof
    print "Lambda(95%) : ", lamda95
    print "Lambda      : ", lambdastart
    print "P-Value     : ", pvalue

else:

    maxval = -1.0 * float("inf")

    if (args.numofcp == 1):
    
        cp1stop = time-1
    
        if args.cp1start <= 0 or args.cp1start > time-1:
            print "CP1 start invalid value"
            exit(1)
    
        if args.cp1stop < 0:
            cp1stop = time-1
        else:
            cp1stop = args.cp1stop
    
        if cp1stop <= args.cp1start or cp1stop > time-1:
            print "CP1 stop invalid value"
            exit(1)
    
        cp = 0
        idx = 0
        for c_p in range(args.cp1start, cp1stop):
            start = tempo.time()
            cstart = tempo.clock()
    
            try:
                L1, L2 = changemod.compute_cps(ms, c_p)
            except changemod.Error:
                print "Oops! error in the main function" 
                exit(1)
    
            if (maxval < L1+L2):
                maxval = L1 + L2
                cp = c_p
        
            fp.write(str(c_p) + " " + str(L1+L2) + "\n")
    
            end = tempo.time()
            cend = tempo.clock()
    
            if args.iterations:
                print "%10d of %10d time (%10.5f s CPU time %10.5f s)"%(idx+1 , cp1stop-args.cp1start, 
                        end - start, cend - cstart)
            else:
                basicutils.progress_bar(idx+1, cp1stop-args.cp1start)
    
            idx = idx + 1 
    
        print ""
        print ""
        print "Change Point: ", cp, " (",maxval, ")"
    
    elif (args.numofcp == 2):
        cp1 = 0
        cp2 = 0
    
        if args.deltacp > 0:
    
           cp1stop = time-1
    
           if args.cp1start <= 0 or args.cp1start > time-1:
               print "CP1 start invalid value"
               exit(1)
          
           if args.cp1stop < 0:
               cp1stop = time-1
           else:
               cp1stop = args.cp1stop
          
           if cp1stop <= args.cp1start or cp1stop > time-1:
               print "CP1 stop invalid value"
               exit(1)
    
           tot = 0
           for c_p1 in range(args.cp1start, cp1stop):
               for c_p2 in range(c_p1 + args.deltacp, time-1):
                   tot = tot + 1
    
           idx = 0
           for c_p1 in range(args.cp1start, cp1stop):
               for c_p2 in range(c_p1 + args.deltacp, time-1):
                   start = tempo.time()
                   cstart = tempo.clock()
    
                   try:
                       L1, L2, L3 = changemod.compute_cps(ms, c_p1, False, c_p2)
                   except changemod.Error:
                       print "Oops! error in the main function" 
                       exit(1)
     
                   if (maxval < L1+L2+L3):
                       maxval = L1 + L2 + L3
                       cp1 = c_p1
                       cp2 = c_p2
    
                   end = tempo.time()
                   cend = tempo.clock()
    
                   if args.iterations:
                       print "%10d of %10d time (%10.5f s CPU time %10.5f s)"%(idx+1 , tot, 
                            end - start, cend - cstart)
                   else:
                       basicutils.progress_bar(idx+1, tot)
    
                   idx = idx + 1 
     
                   fp.write(str(c_p1) + " " + str(c_p2) + " " 
                           + str(L1+L2+L3) + "\n")
                   
           print ""
           print ""
           print "Change Point: ", cp1, " , ", cp2, " (",maxval, ")"
    
        else:
    
           cp1stop = time-1
    
           if args.cp1start <= 0 or args.cp1start > time-1:
               print "CP1 start invalid value"
               exit(1)
          
           if args.cp1stop < 0:
               cp1stop = time-1
           else:
               cp1stop = args.cp1stop
          
           if cp1stop <= args.cp1start or cp1stop > time-1:
               print "CP1 stop invalid value"
               exit(1)
    
           cp2stop = time-1
    
           if args.cp2start <= 0 or args.cp2start > time-1:
               print "CP2 start invalid value"
               exit(1)
          
           if args.cp2stop < 0:
               cp2stop = time-1
           else:
               cp2stop = args.cp2stop
          
           if cp2stop <= args.cp2start or cp2stop > time-1:
               print "CP2 stop invalid value"
               exit(1)
    
           if args.cp2start <= args.cp1start:
               print "CP2 CP2 start invalid value"
               exit(1)
     
           tot = 0
           for c_p1 in range(args.cp1start, cp1stop):
               for c_p2 in range(args.cp2start, cp2stop):
                   tot = tot + 1
    
           idx = 0
           for c_p1 in range(args.cp1start, cp1stop):
               for c_p2 in range(args.cp2start, cp2stop):
    
                   start = tempo.time()
                   cstart = tempo.clock()
    
                   try:
                       L1, L2, L3 = changemod.compute_cps(ms, c_p1, False, c_p2)
                   except changemod.Error:
                       print "Oops! error in the main function" 
                       exit(1)
           
                   if (maxval < L1+L2+L3):
                       maxval = L1 + L2 + L3
                       cp1 = c_p1
                       cp2 = c_p2
    
                   end = tempo.time()
                   cend = tempo.clock()
    
                   if args.iterations:
                       print "%10d of %10d time (%10.5f s CPU time %10.5f s)"%(idx+1 , tot, 
                            end - start, cend - cstart)
                   else:
                       basicutils.progress_bar(idx+1, tot)
    
                   idx = idx + 1 
     
                   fp.write(str(c_p1) + " " + str(c_p2) + " " 
                           + str(L1+L2+L3) + "\n")
                   
           print ""
           print ""
           print "Change Point: ", cp1, " , ", cp2 ," (",maxval, ")"
    
    elif (args.numofcp == 3):
        cp1 = 0
        cp2 = 0
        cp3 = 0
    
        if args.deltacp > 0:
    
           cp1stop = time-1
    
           if args.cp1start <= 0 or args.cp1start > time-1:
               print "CP1 start invalid value"
               exit(1)
          
           if args.cp1stop < 0:
               cp1stop = time-1
           else:
               cp1stop = args.cp1stop
          
           if cp1stop <= args.cp1start or cp1stop > time-1:
               print "CP1 stop invalid value"
               exit(1)
    
           tot = 0
           for c_p1 in range(args.cp1start, cp1stop):
               for c_p2 in range(c_p1 + args.deltacp, time-1):
                   for c_p3 in range(c_p2 + args.deltacp, time-1):
                       tot = tot + 1
    
           idx = 0
           for c_p1 in range(args.cp1start, cp1stop):
               for c_p2 in range(c_p1 + args.deltacp, time-1):
                   for c_p3 in range(c_p2 + args.deltacp, time-1):
    
                       start = tempo.time()
                       cstart = tempo.clock()
    
                       try:
                           L1, L2, L3, L4 = changemod.compute_cps(ms, c_p1, False, c_p2, c_p3)
                       except changemod.Error:
                           print "Oops! error in the main function" 
                           exit(1)
                       
                       if (maxval < L1+L2+L3+L4):
                           maxval = L1 + L2 + L3 + L4
                           cp1 = c_p1
                           cp2 = c_p2
                           cp3 = c_p3
    
                       end = tempo.time()
                       cend = tempo.clock()
          
                       if args.iterations:
                           print "%10d of %10d time (%10.5f s CPU time %10.5f s)"%(idx+1 , tot, 
                                end - start, cend - cstart)
                       else:
                           basicutils.progress_bar(idx+1, tot)
                       
                       idx = idx + 1 
                       
                       fp.write(str(c_p1) + " " + str(c_p2) + " " 
                               + str(c_p3) + " " 
                               + str(L1+L2+L3+L4) + "\n")
                       
           print ""
           print ""
           print "Change Point: ", cp1, " , ", cp2, " ", cp3, " (",maxval, ")"
    
        else:
    
           cp1stop = time-1
    
           if args.cp1start <= 0 or args.cp1start > time-1:
               print "CP1 start invalid value"
               exit(1)
          
           if args.cp1stop < 0:
               cp1stop = time-1
           else:
               cp1stop = args.cp1stop
          
           if cp1stop <= args.cp1start or cp1stop > time-1:
               print "CP1 stop invalid value"
               exit(1)
    
           cp2stop = time-1
    
           if args.cp2start <= 0 or args.cp2start > time-1:
               print "CP2 start invalid value"
               exit(1)
          
           if args.cp2stop < 0:
               cp2stop = time-1
           else:
               cp2stop = args.cp2stop
          
           if cp2stop <= args.cp2start or cp2stop > time-1:
               print "CP2 stop invalid value"
               exit(1)
    
           if args.cp2start <= args.cp1start:
               print "CP1 CP2 start invalid value"
               exit(1)
    
           cp3stop = time-1
    
           if args.cp3start <= 0 or args.cp3start > time-1:
               print "CP3 start invalid value"
               exit(1)
          
           if args.cp3stop < 0:
               cp3stop = time-1
           else:
               cp3stop = args.cp3stop
          
           if cp3stop <= args.cp3start or cp3stop > time-1:
               print "CP3 stop invalid value"
               exit(1)
    
           if args.cp3start <= args.cp2start:
               print "CP3 CP2 start invalid value"
               exit(1)
     
     
           tot = 0
           for c_p1 in range(args.cp1start, cp1stop):
               for c_p2 in range(args.cp2start, cp2stop):
                   for c_p3 in range(args.cp3start, cp3stop):
                       tot = tot + 1
    
           idx = 0
           for c_p1 in range(args.cp1start, cp1stop):
               for c_p2 in range(args.cp2start, cp2stop):
                   for c_p3 in range(args.cp3start, cp3stop):
    
                       start = tempo.time()
                       cstart = tempo.clock()
    
                       try:
                           L1, L2, L3, L4 = changemod.compute_cps(ms,  
                               c_p1, False, c_p2, c_p3)
                       except changemod.Error:
                           print "Oops! error in the main function" 
                           exit(1)
     
                       if (maxval < L1+L2+L3+L4):
                           maxval = L1 + L2 + L3 + L4
                           cp1 = c_p1
                           cp2 = c_p2
                           cp3 = c_p3
    
                       end = tempo.time()
                       cend = tempo.clock()
          
                       if args.iterations:
                           print "%10d of %10d time (%10.5f s CPU time %10.5f s)"%(idx+1 , tot, 
                                end - start, cend - cstart)
                       else:
                           basicutils.progress_bar(idx+1, tot)
                       
                       idx = idx + 1 
                       
                       fp.write(str(c_p1) + " " + str(c_p2) + " " 
                               + str(c_p3) + " " 
                               + str(L1+L2) + "\n")
                       
           print ""
           print ""
           print "Change Point: ", cp1, " , ", cp2 , " ", cp3, " (",maxval, ")"
    
fp.close()
