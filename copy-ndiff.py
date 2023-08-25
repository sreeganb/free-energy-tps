#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess as sup
import os
import fileinput
import random

nbins = 20
lstart = -2.25
lend = 2.25
binwidth = (lend - lstart)/nbins
print(binwidth*0.20)
bounds = np.zeros(nbins+1)
bounds = np.arange(lstart, lend +binwidth, binwidth)
print(bounds)
lb = np.zeros(np.size(bounds))
ub = np.zeros(np.size(bounds))
for i in range(np.size(bounds)):
    lb[i] = bounds[i] - binwidth*0.20
    ub[i] = bounds[i] + binwidth*0.20
print(lb, ub) 
j = 0
k = 1
#olap_bounds = np.zeros(nbins*2)
olap_bounds = np.zeros(nbins*2+2)
for i in range(np.size(bounds)-1):
    olap_bounds[j] = lb[i]
    olap_bounds[k] = ub[i+1]
    j = j + 2
    k = k + 2
print(olap_bounds)
#--------------------------------------------------------------
j = 0
k = 1
for i in range(nbins):
    sup.run("cp ./new-diff.py ./win_%s"%(i+1), shell = True)
    with fileinput.FileInput("./win_%s/new-diff.py"%(i+1), inplace=True, backup='.bak') as file:
        for line in file:
            print(line.replace('lbound', '%s'%(olap_bounds[j])), end='') 
    with fileinput.FileInput("./win_%s/new-diff.py"%(i+1), inplace=True, backup='.bak') as file:
        for line in file:
            print(line.replace('ubound', '%s'%(olap_bounds[k])), end='') 
    with fileinput.FileInput("./win_%s/new-diff.py"%(i+1), inplace=True, backup='.bak') as file:
        for line in file:
            print(line.replace('blabla', '%s'%(i+1)), end='') 
    j = j + 2
    k = k + 2
