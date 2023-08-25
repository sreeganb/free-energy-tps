#!/usr/bin/env python3.8

import subprocess as sup

start = 5
end = 95
ran = 4

for j in range(ran):
    ibeg = start + j*100
    iend = end + j*100
    for i in range(ibeg,iend):
        sup.run('rm -rf new%s.dcd'%(i+1), shell = True)
        sup.run('rm -rf new%s.vel'%(i+1), shell = True)
