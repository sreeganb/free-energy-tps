#!/usr/bin/env python

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

run = int(input("Enter the index of the run: "))

pp = PdfPages('dist%s.pdf'%(run))

my_palette = ['#355EE7', '#CB201A', '#1E9965', '#D291BC', '#875C36', '#FEB301']
sns.plotting_context(context= "paper", font_scale=1)
sns.set_theme(style="ticks")
sns.set_palette(sns.color_palette(my_palette))

from matplotlib import rcParams
#rcParams['font.family'] = 'sans-serif'
#rcParams['font.sans-serif'] = ['Tahoma']

#plt.rc('font', family='serif', serif='Times')
#plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)
plt.rc('legend',fontsize=8)

width = 3.33
height = width / 1.418
f, ax = plt.subplots()
f.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
#************************************************************************
# Libraries pertaining to MD Analysis
#************************************************************************
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from MDAnalysis.tests.datafiles import (PSF, DCD, PDB, CRD)
from MDAnalysis.analysis import psa
from MDAnalysis.analysis import align
import warnings
#************************************************************************
# suppress some MDAnalysis warnings about writing PDB files
warnings.filterwarnings('ignore') 
#----------------------------------------------------------------------
# Read in the trajectories
#----------------------------------------------------------------------
u1 = mda.Universe('nneutr.psf', 'new%s.dcd'%(run))
#***********************************************************************
# Code block for choosing two atoms, iterating through a trajectory and 
# getting the distances between the atoms along the trajectory
#***********************************************************************
m = int(len(u1.trajectory))
print(m)

def get_dist(atom1, atom2):
    dist1 = distances.dist(atom1, atom2)
    return dist1
"""
Instead of the full trajectory if you require the distances from only a 
particular frame then this should do the trick
"""
disco = []
disoh = []
diff = []
for i in range(m):
    u1.trajectory[i]
    atom1 = u1.select_atoms("resname QM and name C8")
    atom2 = u1.select_atoms("resname QM and name H5")
    dist1 = get_dist(atom1, atom2)
    disco.append(float(dist1[2]))
    atom1 = u1.select_atoms("resname QM and name OH2'")
    atom2 = u1.select_atoms("resname QM and name H5")
    dist1 = get_dist(atom1, atom2)
    disoh.append(float(dist1[2]))
    diff.append(disco[i]-disoh[i])
#diff = disco - disoh
#print(diff)
labels = ["distance"]
dif = pd.DataFrame(diff)
disc = pd.DataFrame(disco)
diso = pd.DataFrame(disoh)
dif.columns = ['distance']
disc.columns = ['distance']
diso.columns = ['distance']
dif.to_csv("hc-oh.csv", columns = ['distance'], index=False)
disc.to_csv("hc.csv", columns = ['distance'], index=False)
diso.to_csv("oh.csv", columns = ['distance'], index=False)
plt.plot(disco, label = r'dis(C8-H5)', linewidth = 1)
plt.plot(disoh, label = r'dis(OH2-H5)', linewidth = 1)
plt.plot(diff, label = r'diff', linewidth = 1)
plt.xlabel(r"Time(fs)")
plt.ylabel(r"Distance ({\AA})")
plt.legend(loc = 'best')
f.set_size_inches(width, height)
pp.savefig()
pp.close()
plt.show()
