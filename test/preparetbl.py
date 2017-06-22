import sys 
import numpy
import pandas
import datetime
import calendar
import scipy.io

sys.path.append("../module")
import basicutils

startingdate = {\
        "Bulgaria"       :(1, 1, 2007), \
        "Czech Republic" :(1, 5, 2004), \
        "Croatia"        :(1, 7, 2013), \
        "Latvia"         :(1, 5, 2004), \
        "Lithuana"       :(1, 5, 2004), \
        "Hungary"        :(1, 5, 2004), \
        "Malta"          :(1, 5, 2004), \
        "poland"         :(1, 5, 2004), \
        "Romania"        :(1, 1, 2007), \
        "Slovenia"       :(1, 5, 2004), \
        "Slovakia"       :(1, 5, 2004)}

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

print mat, dates, cnames

"""
for c, date in startingdate.iteritems():
    print c, date
    for d in dates:
      y0 = int(dates[i][0])
      m0 = int(dates[i][1])
      ld = calendar.monthrange(y0,m0)
"""

tot = 0

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

scipy.io.savemat('data.mat', mdict={'nmat': nmat2})
