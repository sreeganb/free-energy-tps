#!/usr/bin/env python

#------------------------------------------------------------------------------
# Script that analyzes the sampled data from window based TPS simulations and 
# creates the free energy function based on the BOLAS algorithm with aimless
# shooting
# Author: Sree Ganesh
# Date: 31-May-2021
# UPDATE : 07-July-2021
# TODO: Find the free energies in the overlap regions and then use the constants
# to connect the curves and make them continuous so that the whole free energy
# profile can be obtained. 
#------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import subprocess as sup
from matplotlib import rcParams

#rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans',
#                               'Lucida Grande', 'Verdana']
rcParams['font.sans-serif'] = ['Verdana']
rcParams.update({'font.size': 12})

my_palette = ['#355EE7', '#CB201A', '#1E9965', '#D291BC', '#875C36', '#FEB301', '#232323', '#50E4EA']
sns.set(context = "paper", style = "ticks")
sns.set_palette(sns.color_palette(my_palette))

plt.rc('text', usetex = True)
plt.rc('xtick', labelsize = 10)
plt.rc('ytick', labelsize = 10)
plt.rc('axes', labelsize = 10)
width = 3.5
height = width / 1.30
ind = float(input('Do you want to use window probability (1) or cumulative probability (0): '))
shift_curves = int(input('Do you want to shift the overlapping regions (1) or not (0): '))
fit_curves = int(input('Do you want to make a polynomial fit of the data (1) or not (0): '))
#------------------------------------------------------------------------------
# Bin the data within a sampled window and find the free energy function for this
# window 
#------------------------------------------------------------------------------
nbins = 13
nboot = 10
ndegree = 15
for l in range(nboot):
    olap = 0.08
    f, ax = plt.subplots()
    f.set_size_inches(width, height)
    f.subplots_adjust(left = 0.15, bottom = 0.14, right = 0.9, top = 0.9)
    def funclog(freq, nord, filnam):
        kbt_in_kcalmol = 0.5957625
        lfreq = -1.0*kbt_in_kcalmol*np.log(freq)
        df = pd.DataFrame({"orderparam" : nord, "pmf" : lfreq})
        df.to_csv("./overlap_data/%s.csv"%(filnam), index = False)
        return lfreq 
    #------------------------------------------------------------------------------
    def overlap_constants(w1, w1max, w2, w2min, olap):
        # read in the files and find the free energy in the overlap region
        if w1 == 1:
            df1 = pd.read_csv('./overlap_data/%s.csv'%(w1))
            df2 = pd.read_csv('./overlap_data/%s.csv'%(w2))
        else:
            df1 = pd.read_csv('./overlap_data/new%s.csv'%(w1))
            df2 = pd.read_csv('./overlap_data/%s.csv'%(w2))
        ord1 = df1['orderparam']
        free1 = df1['pmf']
        ord2 = df2['orderparam']
        free2 = df2['pmf']
        folap1 = []
        folap2 = []
        for i in range(np.size(ord1)):
            if ord1[i] <= w1max and ord1[i] >= w2min:
                folap1.append(free1[i]) 
            elif ord2[i] <= w1max and ord2[i] >= w2min:
                folap2.append(free2[i])
        shift = np.mean(np.array(folap1)-np.array(folap2))
#        print(shift)
        new2 = df2['pmf'] + shift
        df = pd.DataFrame({"orderparam": ord2, "pmf": new2})
        df.to_csv("./overlap_data/new%s.csv"%(w2), index = False)
        if w1 == 1:
            ax.plot(ord1, free1)
            ax.plot(ord2, new2)
        else:
            ax.plot(ord2, new2)
    #------------------------------------------------------------------------------
    # New smoothening function
    #------------------------------------------------------------------------------
    def match_cons(w1, w1max, w2, w2min, olap):
        if w1 == 1:
            df1 = pd.read_csv('./overlap_data/%s.csv'%(w1))
            df2 = pd.read_csv('./overlap_data/%s.csv'%(w2))
        else:
            df1 = pd.read_csv('./overlap_data/new%s.csv'%(w1))
            df2 = pd.read_csv('./overlap_data/%s.csv'%(w2))
        ord1 = df1['orderparam']
        free1 = df1['pmf']
        ord2 = df2['orderparam']
        free2 = df2['pmf']
        folap1 = []
        folap2 = []
        iparam = 3.0
        for i in range(np.size(ord1)):
            if ord1[i] <= w1max and ord1[i] >= w2min:
                folap1.append(free1[i]) 
            elif ord2[i] <= w1max and ord2[i] >= w2min:
                folap2.append(free2[i])
        if (w1max < iparam):
            shift = folap1[0] - folap2[0]
            print("shift is: ", shift)
            print("border elements are: ", folap1[0], folap2[0])
            new2 = df2['pmf'] + shift
            df = pd.DataFrame({"orderparam": ord2, "pmf": new2})
            df.to_csv("./overlap_data/new%s.csv"%(w2), index = False)
        else:
            n1 = np.size(folap1)
            n2 = np.size(folap2)
            shift = folap1[n1-1] - folap2[n2-1]
            print("shift is: ", shift)
            print("border elements are: ", folap1[n1-1], folap2[n2-1])
            new2 = df2['pmf'] + shift
            df = pd.DataFrame({"orderparam": ord2, "pmf": new2})
            df.to_csv("./overlap_data/new%s.csv"%(w2), index = False)
            
        if w1 == 1:
            ax.plot(ord1, free1)
            ax.plot(ord2, new2)
        else:
            ax.plot(ord2, new2)
    #------------------------------------------------------------------------------
    def plotfree(dset, nbins, dmin, dmax, c, dall, filnam, npoints, shift_curves):
        freq, ordpar = np.histogram(dset, bins = nbins, range = (dmin, dmax), density = False) # make bins and collect the frequencies
        if c :
            nfreq = freq/npoints # Divide the frequencies in each bin with total number of points in the simulation
        nord = (ordpar[1:] + ordpar[:-1])/2.0
        if shift_curves:
            fdat = funclog(nfreq, nord, filnam)
        else:
            fdat = funclog(nfreq, nord, filnam)
            ax.plot(nord, fdat)
        plt.axvline(x = dmin, linewidth = 0.5)
        plt.axvline(x = dmax, linewidth = 0.5)
        return 1
    #------------------------------------------------------------------------------
    # New function for polynomial fitting of the free energy data
    #------------------------------------------------------------------------------
    def polyfit(nwins, l, ndegree):
        li = []
        for i in range(nwins):
            if i == 0:
                dat1 = pd.read_csv('./overlap_data/1.csv')
            else:
                dat1 = pd.read_csv('./overlap_data/new%s.csv'%(i+1))
            li.append(dat1)
        frame = pd.concat(li, axis = 0, ignore_index = True)
        frame = frame[np.isfinite(frame).all(1)]
        print(frame["orderparam"], frame["pmf"])
        coeffs = np.poly1d(np.polyfit(frame["orderparam"], frame["pmf"], ndegree))
        print(coeffs)
        frame.to_csv('./sample-pmfs/final-shift-data_%s.csv'%(l), index = False)
        xp = np.linspace(-1.6, 1.6, 400)
        ax.scatter(frame["orderparam"], frame["pmf"], facecolors = 'None')
        ax.plot(xp, coeffs(xp))
#    newdset = []
    #---------------------------------------------------------------------------
    # Set some parameters same as in new-ind-fenergy.py
    #---------------------------------------------------------------------------
    n_wins = 15
    lstart = -1.6
    lend = 1.6 
    binwidth = (lend - lstart)/n_wins
    bounds = np.zeros(n_wins+1)
    bounds = np.arange(lstart, lend+binwidth, binwidth)
    lb = np.zeros(np.size(bounds))
    ub = np.zeros(np.size(bounds))
    for i in range(np.size(bounds)):
        lb[i] = bounds[i] - binwidth*0.20
        ub[i] = bounds[i] + binwidth*0.20
    j = 0
    k = 1
    wins = np.zeros(n_wins*2)
    for i in range(np.size(bounds)-1):
        wins[j] = lb[i]
        wins[k] = ub[i+1]
        j = j + 2
        k = k + 2
    newdset = []
    nums = np.arange(1, n_wins+1)
    print("nbins = ", nbins)    
#    wins = np.array([-3.04, -2.76, -2.84, -2.56, -2.64, -2.36, -2.44, -2.16, -2.24, -1.96, -2.04, -1.76, -1.84, -1.56, -1.64, -1.36, -1.44, -1.16, -1.24, -0.96, -1.04, -0.76, -0.84, -0.56, -0.64, -0.36, -0.44, -0.16, -0.24, 0.04, -0.04, 0.24, 0.16, 0.44, 0.36, 0.64, 0.56, 0.84, 0.76, 1.04, 0.96, 1.24, 1.16, 1.44, 1.36, 1.64, 1.56, 1.84, 1.76, 2.04, 1.96, 2.24, 2.16, 2.44, 2.36, 2.64, 2.56, 2.84, 2.76, 3.04])
#   nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    if ind:
        j = 0
        for i in range(np.size(nums)):
            #-----------------------------------------------------------------
            #Create a new variable called dold which will read the whole file
            # from this choose only a subset of data points and pass it on to
            # variable d, keeping everything else the same. 
            #-----------------------------------------------------------------
            dold = np.loadtxt('./data/%s.csv'%(nums[i]))
            d = np.random.choice(dold, 20000)
            k = plotfree(d, nbins, wins[j], wins[j+1], 1, newdset, nums[i], np.size(d), shift_curves)
            print(wins[j], wins[j+1])
            j = j + 2
    
    j = 0
    if shift_curves:
        for i in range(np.size(nums)-1):
            fini = match_cons(nums[j], wins[2*j+1] , nums[j+1], wins[2*j+2], olap)
            j = j + 1
    
    if fit_curves:
        polyfit(np.size(nums), l, ndegree)    
    ax.set_ylabel('Free energy (Kcal/mol)')
    ax.set_xlabel('Order parameter (dis(NC) - dis(OC))')
    plt.axvline(x=0.0, color = 'r')
