#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess as sup
import os
import fileinput
import random

nbins = 15
lstart = -1.6
lend = 1.6
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
# Function to find values of CH and OH distances that correspond
# to physical situations and not some unphysical point in the 
# phase space
#--------------------------------------------------------------
def find_val(wmin, wmax):
    diff = np.zeros(501)
    hcvals = np.zeros(2)
    ohvals = np.zeros(2)
    diff = np.loadtxt("./parent-traj/hc-oh.csv", dtype="float")
    hc = np.loadtxt("./parent-traj/hc.csv", dtype="float")
    oh = np.loadtxt("./parent-traj/oh.csv", dtype="float")
    midpt = (wmin+wmax)/2.0
    somept = wmin + 0.008
    ordclosest = diff - somept
    hcvals[0] = hc[np.argmin(np.absolute(ordclosest))]
    ohvals[0] = oh[np.argmin(np.absolute(ordclosest))] 
    somept = wmax - 0.008
    ordclosest = diff - somept
    hcvals[1] = hc[np.argmin(np.absolute(ordclosest))]
    ohvals[1] = oh[np.argmin(np.absolute(ordclosest))]
    return hcvals[1], ohvals[1]
#--------------------------------------------------------------
j = 0
k = 1
for i in range(nbins):
    sup.run("mkdir win_%s"%(i+1), shell = True)
    sup.run("mkdir ./win_%s/results"%(i+1), shell = True)
    sup.run("mkdir ./win_%s/react-dis"%(i+1), shell = True)
    sup.run("mkdir ./win_%s/non-reac-dis"%(i+1), shell = True)
    sup.run("mkdir ./win_%s/output-files"%(i+1), shell = True)
    sup.run("cp diff.py ./win_%s/react-dis"%(i+1), shell = True)
    sup.run("cp remove-dcd.py ./win_%s/results"%(i+1), shell = True)
    sup.run("cp submit_mpi.input ./win_%s"%(i+1), shell = True)
    sup.run("cp ./sample-files/* ./win_%s"%(i+1), shell = True)
    sup.run("cp ./new-diff.py ./win_%s"%(i+1), shell = True)
    with open("./win_%s/README"%(i+1), 'w') as rdmefil:
        rdmefil.write("window:\n %s : %s"%(olap_bounds[j],olap_bounds[k]))
    with fileinput.FileInput("./win_%s/react-dis/diff.py"%(i+1), inplace=True, backup='.bak') as file:
        for line in file:
            print(line.replace('lbound, ubound', '%s, %s'%(olap_bounds[j], olap_bounds[k])), end='')
    with fileinput.FileInput("./win_%s/submit_mpi.input"%(i+1), inplace=True, backup='.bak') as file:
        for line in file:
            print(line.replace('lbound ubound', '%s %s'%(olap_bounds[j], olap_bounds[k])).replace('winnum', '%s'%(i+1)), end='')
    hcmax, ohmin = find_val(olap_bounds[j], olap_bounds[k])
    with fileinput.FileInput("./win_%s/reaction.str"%(i+1), inplace=True, backup='.bak') as file:
        for line in file:
            print(line.replace('mxxxhc', '%s'%(hcmax)), end='')
    with fileinput.FileInput("./win_%s/reaction.str"%(i+1), inplace=True, backup='.bak') as file:
        for line in file:
            print(line.replace('minnoh', '%s'%(ohmin)), end='')
    with fileinput.FileInput("./win_%s/regular-tps.inp"%(i+1), inplace=True, backup='.bak') as file:
        for line in file:
            print(line.replace('wwmi', '%s'%(olap_bounds[j])), end='') 
    with fileinput.FileInput("./win_%s/regular-tps.inp"%(i+1), inplace=True, backup='.bak') as file:
        for line in file:
            print(line.replace('wwma', '%s'%(olap_bounds[k])), end='') 
    randseed = random.randint(1000, 10000000)
    with fileinput.FileInput("./win_%s/regular-tps.inp"%(i+1), inplace=True, backup='.bak') as file:
        for line in file: 
            print(line.replace('ranseed', '%s'%(randseed)), end='') 
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
