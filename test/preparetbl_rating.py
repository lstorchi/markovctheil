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

    nametonum = dict((v,k) for k,v in enumerate(calendar.month_abbr))

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

def dump_file (count, values):

     sep = []
     moody = []
     fitch = [] 
     
     for v in values:
         if len(v) == 3:
           if v[0] == "S&P":
               sep.append([v[1], v[2]])
           elif v[0] == "Moody\'s":
               moody.append([v[1], v[2]])
           elif v[0] == "Fitch":
               fitch.append([v[1], v[2]])
         elif len(v) == 4:
           if v[0] == "S&P":
               sep.append([v[1], v[3]])
           elif v[0] == "Moody\'s":
               moody.append([v[1], v[3]])
           elif v[0] == "Fitch":
               fitch.append([v[1], v[3]])
     
     vsep = basic_repter (sep)
     vmoody = basic_repter (moody)
     vfitch = basic_repter (fitch)
     
     outfilename = count+"_data.mat"
     
     if os.path.exists(outfilename):
         print "File ", outfilename, " exist, removing it "
         os.remove(outfilename)
     
     datadict = mdict={'sep': vsep , 'moody': vmoody , 'fitch': vfitch}
     
     scipy.io.savemat(outfilename, datadict)

#####################################################################

file = ""

if len(sys.argv) == 2:
    file = sys.argv[1]
else:
    print "Usage ", sys.argv[0] , " filename "
    exit(1)

df = pandas.read_excel(file, header=[0, 1])

#print the column names
cn = df.columns

countries = set()
for v in cn:
    countries.add(v[0])

for c in countries:
    vals = df[c].values
    #print vals
    dump_file (c, vals)

