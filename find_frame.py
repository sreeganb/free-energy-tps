#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import subprocess as sup

diff = np.zeros(501)
diff = np.loadtxt("./parent-traj/hc-oh.csv", dtype="float")
hc = np.loadtxt("./parent-traj/hc.csv", dtype="float")
oh = np.loadtxt("./parent-traj/oh.csv", dtype="float")
index = int(input("Enter the index of the window: "))
#wmin = float(input("Enter the minimum for the window: "))
#wmax = float(input("Enter the maximum for the window: "))

#------------------------------------------------------------------
# Read the OH and HC distances and find out the minimum and maximum 
# for each window
#------------------------------------------------------------------
#oh = np.loadtxt('./win_%s/README'%(index))
with open("./win_%s/README"%(index)) as fil:
    a = fil.readlines()
    line = str(a[-1]).split()
print("Lower: ", line[0], "Upper: ", line[2])
wmin = float(line[0])
wmax = float(line[2])
midpt = (wmin+wmax)/2.0
#somept = wmax - 0.08
somept = wmin + 0.008
ordclosest = diff - somept
print("Closest to the minimum side: ")
print(min(np.absolute(ordclosest)))
print(np.argmin(np.absolute(ordclosest)))
print("value of HC for index: ", hc[np.argmin(np.absolute(ordclosest))])
print("value of OH for index: ", oh[np.argmin(np.absolute(ordclosest))])

somept = wmax - 0.008
ordclosest = diff - somept
print("Closest to the maximum side: ")
print(min(np.absolute(ordclosest)))
print(np.argmin(np.absolute(ordclosest)))
print("value of HC for index: ", hc[np.argmin(np.absolute(ordclosest))])
print("value of OH for index: ", oh[np.argmin(np.absolute(ordclosest))])
