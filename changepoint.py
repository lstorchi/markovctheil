import argparse
import numpy
import time as tempo
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

parser.add_argument("--delta-cp", help="DElta time between CPs (default: 0) " + 
        "if delta > 0 cp2 and cp3 start and stop values wont be used ", \
        type=int, required=False, default=0, dest="deltacp")


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

        L1, L2 = changemod.compute_cps(ms, errmsg, c_p)
                
        if (L1 == None):
            print errmsg
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

    if args.deltacp  > 0:

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

               L1, L2, L3 = changemod.compute_cps(ms, errmsg, c_p1, c_p2)
                   
               if (L1 == None):
                   print errmsg
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

               L1, L2, L3 = changemod.compute_cps(ms, errmsg, c_p1, c_p2)
                   
               if (L1 == None):
                   print errmsg
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

    if args.deltacp  > 0:

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

                   L1, L2, L3, L4 = changemod.compute_cps(ms, errmsg, c_p1, c_p2, c_p3)
                       
                   if (L1 == None):
                       print errmsg
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

                   L1, L2, L3, L4 = changemod.compute_cps(ms, errmsg, 
                           c_p1, c_p2, c_p3)
                       
                   if (L1 == None):
                       print errmsg
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
