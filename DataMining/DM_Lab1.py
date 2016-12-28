import pandas as pd
import math

#firts lab DM "preprocessing"

#read Glass.data file
from_glass = pd.read_csv('glass.data', sep=',', header=None, names=None)
print from_glass

#choose correct attributes
from_glass = from_glass.loc[:,1:9]
print from_glass

#calculate expected value
mean = from_glass.mean()
print "Mean value"
print mean

#calculate mediana
print "Median"
median = from_glass.median()
print median

#1/2 min+max
n=[]
for m in range(1,10):
    x = from_glass[m]
    temp = (x.min()+x.max())/2
    n.append(temp)
print "1/2 min+max"
print n

#calculate SD
SD=[]
for m in range(1,10):
    x = from_glass[m]
    x=x-mean[m]
    x = x*x
    x = x.sum()/214
    x=math.sqrt(x)
    SD.append(x)
print "SD"
print SD

#average module variations
n=[]
for m in range(1,10):
    x = from_glass[m]
    x=x-median[m]
    x = abs(x)
    x = x.sum()/214
    n.append(x)
print "average module variations"
print n

#amplitude
n=[]
for m in range(1,10):
    x = from_glass[m]
    temp = (x.max()-x.min())/2
    n.append(temp)
print "1/2 max-min"
print n

#dispersion
print "dispersion"
print from_glass.var()

#min and max values
print "min and max values"
min = from_glass.min()
max = from_glass.max()
print min
print max

#valuation and balance
glass = from_glass
for u in range(1, 10):
    glass[u] = (glass[u]-mean[u])/SD[u-1]
print "valuation and balance"
print glass

#gipercoub
glass = from_glass
for u in range(1, 10):
    glass[u] = (glass[u]-glass[u].min())/(glass[u].max()-glass[u].min())
print "gipercoub"
print glass