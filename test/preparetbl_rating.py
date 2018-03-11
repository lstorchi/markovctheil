import sys 
import numpy
import pandas
import datetime
import calendar
import scipy.io
import os.path

sys.path.append("../module")
import basicutils

DFINAL = 31
MFINAL = 1
YFINAL = 2018

DINIT = 23
MINIT = 11
YINIT = 1998

#####################################################################

def storeandwritemat (name, datadicts, final):

   sepmtx = []
   for d in datadicts:
       l = len(d[name])
       if l < final:
           print "Error in dimension"
   
       row = []
       for j in range(final):
           row.append(d[name][j])
       sepmtx.append(row)
   
   if os.path.exists(name + ".mat"):
        print "File sep.mat exist, removing it "
        os.remove(name + ".mat")
   
   dsepmtx = {name : sepmtx}
   scipy.io.savemat(name+".mat", dsepmtx)

#####################################################################

def basic_repter (values):

    val = [] 

    rat2num = {"AAA": 1, "Aaa":1, "AA": 2, "Aa":2, "AA-": 2, "AA+": 2, \
            "Aa1":2, "Aa2":2, "Aa3":2, "A+":3, "A":3,  "A-":3, \
            "A1":3, "A2":3, "A3":3, "BBB+":4, "BBB":4, "BBB-":4, \
            "Baa1":4, "Baa2":4, "Baa3":4, "BB+":5, "BB":5, "BB-":5, \
            "Ba1":5, "Ba2":5, "Ba3":5, "B+":6, "B":6, "B-":6, "B1":6, \
            "B2":6, "B3":6, "CCC+":7, "CCC":7, "CCC-":7, "CC":7, "C":7, \
            "Caa1":7, "Caa2":7, "Caa3":7, "Ca":7,"C-":7, "SD":8, "D":8, "RD":8}

    nametonum = dict((v,k) for k,v in enumerate(calendar.month_abbr))

    m2 = MFINAL 
    d2 = DFINAL
    y2 = YFINAL
    for i in range(0, len(values)):
        date1 = values[i][1]
        vdate1 = date1.split()
        m1 = nametonum[vdate1[0]]
        d1 = int(vdate1[1])
        y1 = int(vdate1[2])

        da1 = datetime.date(y1, m1, d1)
        da2 = datetime.date(y2, m2, d2)
        delta = da2 - da1
        for j in range(delta.days): # -1 is really neeeded
            val.append(rat2num[values[i][0]])

        m2 = m1
        d2 = d1
        y2 = y1

    return val

#####################################################################

def dump_file (count, values, verbose = False):

     sep = []
     moody = []
     fitch = [] 
     
     tododsep = True
     tododmoody = True
     tododfitch = True

     nametonum = dict((v,k) for k,v in enumerate(calendar.month_abbr))
     
     print "Running ... " + count

     for v in values:
         if (v[0] == "S&P" ) or (v[0] == "Moody\'s") or \
                 (v[0] == "Fitch"):
            date = ""
            
            if len(v) == 3:
                date = v[2]
            elif len(v) == 4:
                date = v[3]
            
            vdate = date.split()
            m = nametonum[vdate[0]]
            d = int(vdate[1])
            y = int(vdate[2])
            
            if (y < 1998):
                date = "Jan 01 1998"
            
            if v[0] == "S&P":
                if tododsep:
                  sep.append([v[1], date])
            elif v[0] == "Moody\'s":
                if tododmoody:
                  moody.append([v[1], date])
            elif v[0] == "Fitch":
                if tododfitch:
                  fitch.append([v[1], date])
            
            if (y < 1998):
                if v[0] == "S&P":
                    tododsep = False
                elif v[0] == "Moody\'s":
                    tododmoody = False
                elif v[0] == "Fitch":
                    tododfitch = False
     
     #print c 
     if verbose:
       print "S&P"
       print sep
     vsep = basic_repter (sep)
     #print vsep
     if verbose:
       print "Moody\'s"
       print moody
     vmoody = basic_repter (moody)
     #print vmoody
     if verbose:
       print "Fitch"
       print fitch
     vfitch = basic_repter (fitch)

     outfilename = count+"_data.mat"

     # remove 
     avsep = []
     for i in range(len(vsep)-324):
         avsep.append(vsep[i])

     avfitch = []
     for i in range(len(vfitch)-179):
         avfitch.append(vfitch[i]) 
    
     datadict = mdict={'sep': avsep , 'moody': vmoody , 'fitch': avfitch}

     return outfilename, datadict

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

print cn

order = [\
    'Belgium', \
    'Bulgaria', \
    'Czech Republic', \
    'Denmark', \
    'Germany', \
    'Ireland', \
    'Greece', \
    'Spain', \
    'France', \
    'Croatia', \
    'Italy', \
    'Latvia', \
    'Lithuania', \
    'Luxembourg', \
    'Hungary', \
    'Malta', \
    'Netherlands', \
    'Austria', \
    'Poland', \
    'Portugal', \
    'Romania', \
    'Slovenia', \
    'Slovakia', \
    'Finland', \
    'Sweden', \
    'UK']

countries = set()
for v in cn:
    countries.add(v[0])

ordercountries = []
for c1 in order:
    for c2 in countries:
        if c2.find(c1) != -1:
            ordercountries.append(c2)


filenames = []
datadicts = []

for c in ordercountries:
    vals = df[c].values
    outfilename, datadict = dump_file (c, vals)
    
    filenames.append(outfilename)
    datadicts.append(datadict)

maxsep = 0
maxmoody = 0
maxfitch = 0
for d in datadicts:
    if len(d["sep"]) > maxsep:
        maxsep = len(d["sep"])
    if len(d["moody"]) > maxmoody:
        maxmoody = len(d["moody"])
    if len(d["fitch"]) > maxfitch:
        maxfitch = len(d["fitch"])

print "Max S&P ", maxsep, "Max Moodys ", maxmoody, "Max Fitch ", maxfitch

realmax = max(maxsep, max(maxmoody, maxfitch))

for d in datadicts:
    l = len(d["sep"])
    if l < realmax:
        v = d["sep"][l-1]
        for j in range(realmax-l):
            d["sep"].append(v)

    l = len(d["moody"])
    if l < realmax:
        v = d["moody"][l-1]
        for j in range(realmax-l):
            d["moody"].append(v)

    l = len(d["fitch"])
    if l < realmax:
        v = d["fitch"][l-1]
        for j in range(realmax-l):
            d["fitch"].append(v)


da1 = datetime.date(YFINAL, MFINAL, DFINAL)
da2 = datetime.date(YINIT, MINIT, DINIT)
delta = da1 - da2

final = delta.days + 1

storeandwritemat ("sep", datadicts, final)
storeandwritemat ("moody", datadicts, final)
storeandwritemat ("fitch", datadicts, final)

for f in filenames:
    print f

exit(1)

for i in range(len(filenames)):

    if os.path.exists(filenames[i]):
         print "File ", filenames[i], " exist, removing it "
         os.remove(filenames[i])
 
    scipy.io.savemat(filenames[i], datadicts[i])
