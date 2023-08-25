#!/usr/bin/env python3.8

#----------------------------------------------------------------------
# Run short TPS simulations within some windows of order parameter
# This is going to pick a random point from all the available points,
# and then run TPS, then when successful will run more such simulations.
# Success is given by the condition, are there any snapshots within the
# window. 
#----------------------------------------------------------------------

import numpy as np
#import pandas as pd
#import seaborn as sns
import subprocess as sup # for executing bash commands
from shutil import copyfile
import re
import glob
import math
import random
import sys # to deal with command line arguments

np.set_printoptions(precision = 6) # precision for printing floating point numbers
print(sys.argv[1:])
minpar = float(sys.argv[1])
maxpar = float(sys.argv[2])
nruns = int(sys.argv[3])
run = int(sys.argv[4])
nreac = int(sys.argv[5])
#----------------------------------------------------------------------
# Some user input variables
#----------------------------------------------------------------------
#minpar = float(input("Enter the lower limit of this window: ") or "0.16")
#maxpar = float(input("Enter the upper limit of this window: ") or "0.44")
#nruns = int(input("Enter the total number of runs: ") or "10") 
#run = int(input("Enter the index of the current trajectory") or "2") # initial trajectory
#nreac = int(input("Enter the index of the next trajectory") or "3") 
for p in range(nruns):
#----------------------------------------------------------------------
# Check if there are less than 5 points within the window in the current
# trajectory. This makes the probability of getting another reactive trajectory at
# 5/41 which is less than 0.1. Thats very small odds and would take a lot of time
# Instead there will be a file now which is going to keep track of how the shooting
# point was chosen and adjust the probabilities accordingly, somehow. 
#----------------------------------------------------------------------
    if run == 0:
        hcdis = np.loadtxt('../parent-traj/hc.csv') 
        ohdis = np.loadtxt('../parent-traj/oh.csv') 
    else: 
        hcdis = np.loadtxt('./react-dis/dis_hc%s.dat'%(run)) 
        ohdis = np.loadtxt('./react-dis/dis_oh%s.dat'%(run))
    diff = hcdis - ohdis
    npoints = 0
    ind = []
    for i in range(len(diff)): 
        if (minpar <= diff[i] <= maxpar):
            npoints = npoints + 1
            ind.append(i+1)
    if npoints < 3:
        if (min(ind) - max(ind) == 0):
            if max(ind) == 21:
                ibeg = 10
                iend = 21
                prob = (21-10+1)/21.0
            else:
                ibeg = min(ind) - 1
                iend = min(ind) + 1
                prob = 3 / 21.0
        else:
            ibeg = min(ind)
            iend = max(ind)
            prob = npoints / 21.0
    else:
        ibeg = 1
        iend = 21 
        prob = 1.0
          
    with open('run_tracker.txt', 'a') as rtrack:
        rtrack.write(' run : %s \n no. of points within the window : %s \n shooting range : %s - %s \n weight for the configurations : %s \n'%(run+1, npoints, ibeg, iend, prob))
    #----------------------------------------------------------------------
    # Choose the shooting point
    #----------------------------------------------------------------------
    #for i in range(np.size(ord_param)):
    #    j = shoot_point(ord_param[0,i])
    #    if j:
    #        indmat.append(i)
    #        win_ordpar.append(float(ord_param[0,i]))
    #        k = k + 1
    print("RUN: ", run)
    #----------------------------------------------------------------------
    # Now comes the hard part. Run the CHARMM calculations
    #----------------------------------------------------------------------
    tpsdumfil = open("./reg-set_tps.inp", "r")
    tpsfil = open("./regular-tps.inp", "w+")
    orig_words = ("ibeg", "iend", "rrr", "nnn", "mruns", "wwmi", "wwma", "ranseed")
    maxrun = nreac + 75 
    randseed = random.randint(1000, 10000000) 
    rep_words = (str(ibeg), str(iend), str(run), str(nreac), str(maxrun), str(minpar), str(maxpar), str(randseed))
    for line in tpsdumfil:
        for orig, rep in zip(orig_words, rep_words):
            line = line.replace(orig, rep)
        tpsfil.write(line)
    tpsdumfil.close()
    tpsfil.close()
    with open("progress.out", "a") as pfile:
        pfile.write(" run : %s \n"%(str(run)))
    pfile.close()
    # Move some files around and then run CHARMM
    #sup.run('mv ./dis_* ./react-dis/', shell = True)
    #sup.run('charmm42sp < regular-tps.inp > output-files/win1_%s.out'%(run), shell = True) 
    sup.run('srun -n 12 /home/u18/antoniou/local/bin/charmm47sp -i regular-tps.inp -o output-files/traj_%s.out'%(run), shell = True) 

    #----------------------------------------------------------------------
    # copy the distances with right names into the right directory
    #----------------------------------------------------------------------
    sup.run('mv dis_hc*.dat react-dis/dis_hc%s.dat'%(nreac), shell = True)
    sup.run('mv dis_oh*.dat react-dis/dis_oh%s.dat'%(nreac), shell = True)
    sup.run('mv *_hc.dat non-reac-dis/', shell = True) 
    sup.run('mv *_oh.dat non-reac-dis/', shell = True) 
    sup.run('mv new.dcd results/new%s.dcd'%(nreac), shell = True) 
    sup.run('mv new.vel results/new%s.vel'%(nreac), shell = True)
    # Increment the runs
    run = run + 1
    nreac = nreac + 1

