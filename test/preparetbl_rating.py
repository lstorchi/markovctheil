import sys 
import numpy
import pandas
import datetime
import calendar
import scipy.io
import os.path

sys.path.append("../module")
import basicutils

#####################################################################

def basic_repter (sep):

    val = [] 

    for i in range(1, len(sep)):
        date1 = sep[i-1][1]
        vdate1 = date1.split()
        m1 = nametonum[vdate1[0]]
        d1 = int(vdate1[1])
        y1 = int(vdate1[2])

        date2 = sep[i][1]
        vdate2 = date2.split()
        m2 = nametonum[vdate2[0]]
        d2 = int(vdate2[1])
        y2 = int(vdate2[2])
        
        d1 = datetime.date(y1, m1, d1)
        d2 = datetime.date(y2, m2, d2)
        delta = d1 - d2 
        for j in range(delta.days-1): # -1 is really neeeded
            val.append(sep[i-1][0])


    return val

#####################################################################

file = ""

if len(sys.argv) == 2:
    file = sys.argv[1]
else:
    print "Usage ", sys.argv[0] , " filename "
    exit(1)

df = pandas.read_excel(file)

#print the column names
cn = df.columns

nametonum = dict((v,k) for k,v in enumerate(calendar.month_abbr))

sep = []
moody = []
fitch = [] 

for i, v in  enumerate(df[cn[0]].values):
    if v == "S&P":
        sep.append([df[cn[1]].values[i], df[cn[2]].values[i]])
    elif v == "Moody\'s":
        moody.append([df[cn[1]].values[i], df[cn[2]].values[i]])
    elif v == "Fitch":
        fitch.append([df[cn[1]].values[i], df[cn[2]].values[i]])

vsep = basic_repter (sep)
vmoody = basic_repter (moody)
vfitch = basic_repter (fitch)

outfilename = "data.mat"

if os.path.exists(outfilename):
    print "File ", outfilename, " exist, removing it "
    os.remove(outfilename)

datadict = mdict={'sep': vsep , 'moody': vmoody , 'fitch': vfitch}

scipy.io.savemat(outfilename, datadict)
