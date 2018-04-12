from bs4 import BeautifulSoup as bs
from urllib.request import (
    urlopen, urlparse, urlunparse, urlretrieve)
import re
import pandas
import numpy

def nettoTimes(url):
    return times(url,'Netto')

def bruttoTimes(url):
    return times(url,'Brutto')

def times(url, marker):
    soup = bs(urlopen(url))
    page = str(soup)
    indexesStart = [m.start() for m in re.finditer('%s</div>\d\d:\d\d:\d\d' % (marker), page)]
    startingTimes = []
    for i in indexesStart:
        sTime = page[i + len(marker) +6:(i + 14 +len(marker))]
        startingTimes.append(sTime)

    return numpy.asarray(startingTimes)

def lookAllResults():
    url="http://berlinerhm.r.mikatiming.de/2018/?page=%s&event=HML&pid=search"


    t = pandas.DataFrame(columns=['Netto', 'Brutto','NettoInSeconds','BruttoInSeconds'])

    netto = numpy.empty(0)
    brutto = numpy.empty(0)
    nettoSeconds = numpy.empty(0)
    bruttoSeconds = numpy.empty(0)
    for i in range(1300,1330):

        print(i)
        n = nettoTimes(url % (i))
        b = bruttoTimes(url % (i))
        if len(n)!=len(b):
            print("Anzahl Brutto Netto Zeiten stimmt nicht bei index %i" % (i))
            #return
        else:
            nSeconds=calcSeconds(n)
            bSeconds = calcSeconds(b)
            netto=numpy.concatenate((netto, n), axis=0)
            brutto = numpy.concatenate((brutto, b), axis=0)
            nettoSeconds = numpy.concatenate((nettoSeconds, nSeconds), axis=0)
            bruttoSeconds = numpy.concatenate((bruttoSeconds, bSeconds), axis=0)

    t['Netto'] = netto
    t['Brutto'] = brutto
    t['NettoInSeconds']=nettoSeconds
    t['BruttoInSeconds'] = bruttoSeconds
    #print(t)

    t.to_csv('HalbmarathonData/Halbmarathon1300to1329.csv', header=t.columns, sep=',')


def calcSeconds(times):
    seconds=numpy.empty(times.shape)
    index=0;
    for t in times:
        seconds[index]=int(t[1])*60*60 + int(t[3:5])*60+int(t[6:8])
        index+=1
    return seconds

def concateFiles():
    t = pandas.DataFrame(columns=['Netto', 'Brutto', 'NettoInSeconds', 'BruttoInSeconds'])
    data1to99= pandas.read_csv('HalbmarathonData/Halbmarathon1to99.csv', delimiter=',')
    data100to199 = pandas.read_csv('HalbmarathonData/Halbmarathon100to199.csv', delimiter=',')
    data200to298 = pandas.read_csv('HalbmarathonData/Halbmarathon200to298.csv', delimiter=',')
    data299to499 = pandas.read_csv('HalbmarathonData/Halbmarathon299to499.csv', delimiter=',')
    data500to699 = pandas.read_csv('HalbmarathonData/Halbmarathon500to699.csv', delimiter=',')
    data700to899 = pandas.read_csv('HalbmarathonData/Halbmarathon700to899.csv', delimiter=',')
    data900to1099 = pandas.read_csv('HalbmarathonData/Halbmarathon900to1099.csv', delimiter=',')
    data1100to1299 = pandas.read_csv('HalbmarathonData/Halbmarathon1100to1299.csv', delimiter=',')
    data1300to1329 = pandas.read_csv('HalbmarathonData/Halbmarathon1300to1329.csv', delimiter=',')
    t=pandas.concat([data1to99,data100to199,data200to298,data299to499,data500to699,data700to899,data900to1099,data1100to1299,data1300to1329])
    t['Startzeit']=(t['BruttoInSeconds'])- t['NettoInSeconds']
    t.to_csv('HalbmarathonData/HalbmarathonAlmostComplete.csv', header=t.columns, sep=',')

def sortResults():
    data = pandas.read_csv('HalbmarathonData/HalbmarathonAlmostComplete.csv', delimiter=',')
    data= data.sort_values(by=['Startzeit','NettoInSeconds'])
    data.to_csv('HalbmarathonData/HalbmarathonAlmostCompleteSorted.csv', header=data.columns, sep=',')

def findAllOvertaken(startzeit,laufzeitInS):
    data = pandas.read_csv('HalbmarathonData/HalbmarathonAlmostCompleteSorted.csv', delimiter=',')
    earlierStartet=data['Startzeit']<startzeit
    laterAtEnd=data['BruttoInSeconds']>startzeit+laufzeitInS
    overtaken=data[earlierStartet & laterAtEnd]
    print(overtaken.shape)

    faster=data[data['NettoInSeconds']<5700]
    print(faster.shape)
    #overtaken.to_csv('HalbmarathonData/Overtaken.csv', header=overtaken.columns, sep=',')


def calcRunnersPerMinute():
    data = pandas.read_csv('HalbmarathonData/HalbmarathonAlmostCompleteSorted.csv', delimiter=',')
    runners=[]
    for i in range(0,120):
        timeLower=60.0*60.0+i*60.0
        timeUpper=60.0*60.0+(i+1)*60.0
        lower=data['BruttoInSeconds']>=timeLower
        upper=data['BruttoInSeconds']<timeUpper
        r=data[lower & upper ]
        print("Minute %i :%i" % (i+60.0,len(r)))


#findAllOvertaken(913.0,7468.0)
#findAllOvertaken(1802.0,6000)
calcRunnersPerMinute()
#vorne block C ist Start nach 220 Sekunden
#concateFiles()
#sortResults()
#netto=nettoTimes("http://berlinerhm.r.mikatiming.de/2018/?page=2&event=HML&pid=search")
#brutto=bruttoTimes("http://berlinerhm.r.mikatiming.de/2018/?page=2&event=HML&pid=search")

#lookAllResults()
#print(netto)
#print(brutto)

