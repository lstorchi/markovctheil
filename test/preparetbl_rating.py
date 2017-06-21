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

    rat2num = {"AAA": 1, "Aaa":1, "AA": 2, "Aa":2, "AA-": 2, "AA+": 2, \
            "Aa1":2, "Aa2":2, "Aa3":2, "A+":3, "A":3,  "A-":3, \
            "A1":3, "A2":3, "A3":3, "BBB+":4, "BBB":4, "BBB-":4, \
            "Baa1":4, "Baa2":4, "Baa3":4, "BB+":5, "BB":5, "BB-":5, \
            "Ba1":5, "Ba2":5, "Ba3":5, "B+":6, "B":6, "B-":6, "B1":6, \
            "B2":6, "B3":6, "CCC+":7, "CCC":7, "CCC-":7, "CC":7, "C":7, \
            "Caa1":7, "Caa2":7, "Caa3":7, "Ca":7, "SD":8, "D":8, "RD":8}

    nametonum = dict((v,k) for k,v in enumerate(calendar.month_abbr))

    m2 = 5 
    d2 = 31
    y2 = 2017
    for i in range(0, len(sep)):
        date1 = sep[i][1]
        vdate1 = date1.split()
        m1 = nametonum[vdate1[0]]
        d1 = int(vdate1[1])
        y1 = int(vdate1[2])

        da1 = datetime.date(y1, m1, d1)
        da2 = datetime.date(y2, m2, d2)
        delta = da2 - da1
        for j in range(delta.days): # -1 is really neeeded
            val.append(rat2num[sep[i-1][0]])

        m2 = m1
        d2 = d1
        y2 = y1

    return val

#####################################################################

def dump_file (count, values):

     sep = []
     moody = []
     fitch = [] 
     
     tododsep = True
     tododmoody = True
     tododfitch = True

     nametonum = dict((v,k) for k,v in enumerate(calendar.month_abbr))

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
     #print "SeP"
     vsep = basic_repter (sep)
     #print "Mod"
     vmoody = basic_repter (moody)
     #print "fitch"
     vfitch = basic_repter (fitch)
     #print "" 

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
    dump_file (c, vals)

