#!/usr/bin/env /home/sree/anaconda3/bin/python

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

my_palette = ['#355EE7', '#CB201A', '#1E9965', '#D291BC', '#875C36', '#FEB301', '#232323', '#50E4EA']
sns.set(context = "paper", style = "ticks")
sns.set_palette(sns.color_palette(my_palette))

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('glu2TS-fenergy.pdf')
plt.rc('text', usetex = True)
plt.rc('xtick', labelsize = 8)
plt.rc('ytick', labelsize = 8)
plt.rc('axes', labelsize = 8)
width = 3.5
height = width / 1.30
ind = float(input('Do you want to use window probability (1) or cumulative probability (0): '))
shift_curves = int(input('Do you want to shift the overlapping regions (1) or not (0): '))
fit_curves = int(input('Do you want to make a polynomial fit of the data (1) or not (0): '))
choose_row = int(input('Do you want to use a fixed number of rows for all the windows yes(1) or no(0): '))
if choose_row:
    read_rows = int(input('how many rows do you want to read: '))
#------------------------------------------------------------------------------
# Bin the data within a sampled window and find the free energy function for this
# window 
#------------------------------------------------------------------------------
ndegree = 17 # degree of the polynomial fit
nbins = 13
olap = 0.08
f, ax = plt.subplots()
f.set_size_inches(width, height)
f.subplots_adjust(left = 0.15, bottom = 0.14, right = 0.9, top = 0.9)
def funclog(freq, nord, filnam):
    kbt_in_kcalmol = 0.5957625
    #print(freq)
    lfreq = -1.0*kbt_in_kcalmol*np.log(freq)
    df = pd.DataFrame({"orderparam" : nord, "pmf" : lfreq})
    df.to_csv("./overlap_data/%s.csv"%(filnam), index = False)
    return lfreq 
#------------------------------------------------------------------------------
def overlap_constants(w1, w1max, w2, w2min, olap):
    # read in the files and find the free energy in the overlap region
    if choose_row:
        if w1 == 1:
            df1 = pd.read_csv('./overlap_data/%s.csv'%(w1),nrows=read_rows)
            df2 = pd.read_csv('./overlap_data/%s.csv'%(w2),nrows=read_rows)
        else:
            df1 = pd.read_csv('./overlap_data/new%s.csv'%(w1),nrows=read_rows)
            df2 = pd.read_csv('./overlap_data/%s.csv'%(w2),nrows=read_rows)
    else:
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
    #print(shift)
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
#    nfreq, ordpar = np.histogram(dset, bins = nbins, range = (dmin, dmax), density = True) # make bins and collect the frequencies
    if c :
#        nothing = 0
        nfreq = freq/npoints # Divide the frequencies in each bin with total number of points in the simulation
#    else:
#        nfreq = freq/np.size(dall)
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
def polyfit(nwins):
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
    print("ndegree value is: ", ndegree)
    coeffs = np.poly1d(np.polyfit(frame["orderparam"], frame["pmf"], ndegree))
    print(coeffs)
    frame.to_csv('./final-shift-data.csv', index = False)
    xp = np.linspace(-1.6, 1.6, 400)
    ax.scatter(frame["orderparam"], frame["pmf"], facecolors = 'None')
    ax.plot(xp, coeffs(xp))
    #plt.show()
#------------------------------------------------------------------------------
# Create arrays with the bin boundaries to analyze the data 
#------------------------------------------------------------------------------
nwins = 15
lstart = -1.6
lend = 1.6 
binwidth = (lend - lstart)/nwins
bounds = np.zeros(nwins+1)
print("size of bounds:",np.size(bounds))
bounds = np.arange(lstart, lend+binwidth, binwidth)
print("size of bounds:",np.size(bounds))
lb = np.zeros(np.size(bounds))
ub = np.zeros(np.size(bounds))
for i in range(np.size(bounds)):
    lb[i] = bounds[i] - binwidth*0.20
    ub[i] = bounds[i] + binwidth*0.20
j = 0
k = 1
wins = np.zeros(nwins*2)
for i in range(np.size(bounds)-1):
    wins[j] = lb[i]
    wins[k] = ub[i+1]
    print(i, j, k, np.size(bounds))
    j = j + 2
    k = k + 2
newdset = []
nums = np.arange(1, nwins+1)
if ind:
    j = 0
    for i in range(np.size(nums)):
        d = np.loadtxt('./data/%s.csv'%(nums[i]))
        #k = plotfree(d, nbins, np.min(d), np.max(d), 1, newdset)
        k = plotfree(d, nbins, wins[j], wins[j+1], 1, newdset, nums[i], np.size(d), shift_curves)
        print(wins[j], wins[j+1])
        j = j + 2

j = 0
#for i in range(np.size(nums)-1):
if shift_curves:
    for i in range(np.size(nums)-1):
        #print(nums[j], wins[2*j+1], nums[j+1], wins[2*j+2])
        #fin = overlap_constants(nums[j], wins[2*j+1] , nums[j+1], wins[2*j+2], olap)
        print(2*j+1, 2*j+2)
        fini = match_cons(nums[j], wins[2*j+1] , nums[j+1], wins[2*j+2], olap)
        j = j + 1

#else:    
#    #------------------------------------------------------------------------------
#    # For the cumulative probability distribution, we need to split the data
#    # into bins, I suppose and then calculate for each bin or, split the data into
#    # windows and then make histograms within each window and then calculate the 
#    # free energies for each window. 
#    #------------------------------------------------------------------------------
#    def conc_data(d, dnew):
#        d = np.concatenate((d, dnew), axis = 0)
#        return d
#    
#    dall = np.loadtxt('./data/%s.csv'%(nums[0]))
#    nnums = nums[1:]
#    for i in nnums:
#        dnew = np.loadtxt('./data/%s.csv'%(i))
#        dall = conc_data(dall, dnew) 
#    print(np.size(dall))
#    # Separate the cumulative probability distribution into windows and calculate
#    # free energies
#    def chk_win(elem, wmin, wmax):
#        a = 0
#        if elem <= wmax and elem >= wmin:
#            a =1
#        return a
#    k = 0    
#    for j in range(np.size(nums)):
#        if os.path.isfile('cumuldata/%s.csv'%(nums[j])):
#            sup.run('rm -rf cumuldata/%s.csv'%nums[j], shell = True)
#        for i in dall:
#            b = chk_win(i, wins[k], wins[k+1])
#            if b:
#                filord = open('cumuldata/%s.csv'%(nums[j]), 'a')
#                filord.write(str(i) + '\n')
#                filord.close()
#        k = k + 2
#    j = 0
#    for i in nums:
#        dset = np.loadtxt('cumuldata/%s.csv'%(i))
#        print(j, j+1)
#        k = plotfree(dset, nbins, wins[j], wins[j + 1], 0, dall, i)
#        j = j + 2
if fit_curves:
    polyfit(np.size(nums))    
ax.set_ylabel('Free energy (Kcal/mol)')
ax.set_xlabel('Order parameter (dis(HC) - dis(OH))')
plt.axvline(x=-0.016, color = 'r')
plt.title("GLU 2TS free energy")
#plt.tight_layout()
pp.savefig()
pp.close()
plt.show()
