import os
import sys 
import numpy
import pandas
import datetime
import calendar
import scipy.io

sys.path.append("../module")
import basicutils

file

if len(sys.argv) == 2:
    file = sys.argv[1]
else:
    print "Usage ", sys.argv[0] , " filename "
    exit(1)

fp = open(file)

numofd = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

alllines = []
for l in fp:
    alllines.append(l.split())

fp.close()

maincounter = 1
counter = 1
mese = 0
for i in range(len(alllines)):
    if numofd[mese] == counter:
        print maincounter, alllines[i][1], alllines[i][2]
        maincounter += 1
        mese += 1
        counter = 1
        if mese == len(numofd):
            mese = 0

    counter += 1

