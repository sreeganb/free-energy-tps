#!/usr/bin/env python3.8

import subprocess as sup
import os

sup.run('rm ./visualize/data/run_info.dat', shell = True)
for i in range(15):
    os.chdir('win_%s'%(i+1))
    sup.run('ls', shell=True)
    #sup.run('python3.8 data-collect.py', shell=True)
    sup.run('python3.8 new-diff.py', shell=True)
    sup.run('cp unsorted.csv ../visualize/data/%s.csv'%(i+1), shell=True)
    with open('sorted.csv', 'r') as fp:
        x = len(fp.readlines())/21
    with open('../visualize/data/run_info.dat', 'a') as ridat:
        if i == 0:
            ridat.write("window, No. of trajs.\n")
            ridat.write("%s,  %s\n"%(i+1, x))
        else:
            ridat.write("%s,  %s\n"%(i+1, x))
    sup.run('cp window%s.pdf ../visualize/data/'%(i+1), shell=True)
    os.chdir('../')
