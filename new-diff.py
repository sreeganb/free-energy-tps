#!/usr/bin/env python3.8

#----------------------------------------------------------------------
# Script to analyze the short trajectories run with window based 
# TPS ensemble. The TPS ensemble is created with the shooting 
# algorithm. 
#----------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as axi 
import seaborn as sns
import glob
import re
import os

my_palette = ['#355EE7', '#CB201A', '#1E9965', '#D291BC', '#FA8B1D','#875C36', '#FEB301']
sns.set(style = "ticks", context = "paper", font_scale = 1.4)
sns.set_palette(sns.color_palette(my_palette))

#from matplotlib.lines import Lines2D
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('windowblabla.pdf')

f, ax=plt.subplots()
#----------------------------------------------------------------------
# Read in the files from the reactive directory
#path =r'/home/sree/work/bolas_mat2a/new_try/window_m4/react-dis'
path = os.getcwd()
print(path)
# N-C distances 
all_files = sorted(glob.glob(path + "/react-dis/*_hc*.dat"))
print(all_files)
li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=None)
    li.append(df)

hc_dis = pd.concat(li, axis=1, ignore_index=True)
hc1_dis = pd.concat(li, axis=0, ignore_index=True)

# O-C distances 
all_files = sorted(glob.glob(path + "/react-dis/*_oh*.dat"))

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=None)
    li.append(df)

oh_dis = pd.concat(li, axis=1, ignore_index=True)
oh1_dis = pd.concat(li, axis=0, ignore_index=True)
np.savetxt('sorted.csv',(hc1_dis-oh1_dis).sort_values(by=[0],ascending=True),fmt='%10.8f')
np.savetxt('unsorted.csv',(hc1_dis-oh1_dis),fmt='%10.8f')
sorted_array = np.array((hc1_dis-oh1_dis).sort_values(by=[0], ascending=True))
#sns.histplot(sorted_array, bins='auto')
sns.histplot(hc1_dis-oh1_dis, bins='auto')
plt.xlabel("dis(H-C)-dis(O-H)")
plt.legend([], [], frameon=False)
plt.axvline(lbound, color= 'k', linestyle = '--')
plt.axvline(ubound, color= 'k', linestyle = '--')
#axi.Axes.axvline(self,x = 0.5)
pp.savefig()
pp.close()
