import os
import sys 
import numpy
import pandas
import datetime
import calendar
import scipy.io

sys.path.append("../module")
import basicutils

startingdate = {\
        'Belgium'        :(27, 11, 1998), \
        'Bulgaria'       :(27, 11, 1998), \
        'Czech Republic' :(27, 11, 1998), \
        'Denmark'        :(27, 11, 1998), \
        'Germany'        :(27, 11, 1998), \
        'Ireland'        :(27, 11, 1998), \
        'Greece'         :(27, 11, 1998), \
        'Spain'          :(27, 11, 1998), \
        'France'         :(27, 11, 1998), \
        'Croatia'        :(27, 11, 1998), \
        'Italy'          :(27, 11, 1998), \
        'Latvia'         :(27, 11, 1998), \
        'Lithuania'      :(27, 11, 1998), \
        'Luxembourg'     :(27, 11, 1998), \
        'Hungary'        :(27, 11, 1998), \
        'Malta'          :(27, 11, 1998), \
        'Netherlands'    :(27, 11, 1998), \
        'Austria'        :(27, 11, 1998), \
        'Poland'         :(27, 11, 1998), \
        'Portugal'       :(27, 11, 1998), \
        'Romania'        :(27, 11, 1998), \
        'Slovenia'       :(27, 11, 1998), \
        'Slovakia'       :(27, 11, 1998), \
        'Finland'        :(27, 11, 1998), \
        'Sweden'         :(27, 11, 1998), \
        'United Kingdom' :(27, 11, 1998)}



file = ""

if len(sys.argv) == 2:
    file = sys.argv[1]
else:
    print "Usage ", sys.argv[0] , " filename "
    exit(1)

df = pandas.read_excel(file)
cn = df.columns

cnames = []

for name in df[cn[0]].values:
    cnames.append(name.replace(",", " "))

count = len(df[cn[0]].values)

mat = numpy.zeros((count, len(cn)-1))

dates = []

for i in range(1,len(cn)):
    date = cn[i].split("-")
    dates.append(date)

c = 0
for i in range(1,len(cn)):
    values = df[cn[i]].values
    #print values
    j = 0
    for k in range(0, len(values)):
        v = values[k]
        if (basicutils.is_float(v)):
            mat[j,c] = v
        else:
            mat[j,c] = float('nan')
            #print "Error in value ", v
        j = j + 1
    c = c + 1

#print mat, dates, cnames

print cnames

for c, date in startingdate.iteritems():
    print c, date
    m0 = date[1]
    y0 = date[2]
    idx = -1

    for ridx in range(len(cnames)):
        if cnames[ridx].find(c) != -1:
            cidx = -1
            for cidx in range(len(dates)):
                y = int(dates[cidx][0])
                m = int(dates[cidx][1])

                if (y >= y0 ) and (m >= m0):
                    break

            if cidx >= 0:
                for i in range(cidx):
                    mat[ridx, i] = float('nan')
           
            print ridx, cidx
            break

tot = 0

for i in range(0,len(cnames)):
    sys.stdout.write(cnames[i] + " ")
    for j in range(0,mat.shape[1]):
        sys.stdout.write("%5.2f "%(mat[i,j])) 
    sys.stdout.write("\n")

for i in range(0,len(dates)):
    y0 = int(dates[i][0])
    m0 = int(dates[i][1])
    ld = calendar.monthrange(y0,m0)

    tot = tot + ld[1]

nmat = numpy.zeros((count, tot))

for k in range(mat.shape[0]):
    aidx = 0

    for i in range(0,len(dates)):
        y0 = int(dates[i][0])
        m0 = int(dates[i][1])
        ld = calendar.monthrange(y0,m0)

        for j in range(ld[1]):
            nmat[k, aidx] = mat[k,i]
            aidx = aidx + 1

# remove the first column

nmat2 = [ row[1:] for row in nmat ]

outfilename = "data.mat"

if os.path.exists(outfilename):
    print "File ", outfilename, " exist, removing it "
    os.remove(outfilename)

scipy.io.savemat(outfilename, mdict={'nmat': nmat2})
