#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 19:39:25 2018

@author: shaowen1994
"""

import numpy as np
import os

file = open("TM_scores_1.fa","w")
err = open("Failed_PDBS_1.fa","w")

num = int(os.popen('ls *.pdb|wc -l').read())

for i in range(200):
    for j in range(num):
        #print(i,j)
        command = './TMalign '+str(i+1)+'.pdb '+str(j+1)+'.pdb -a'
        output = os.popen(command)
        out = output.read()
        if "(if normalized by length of Chain_1)" in out:
            k = out.index("(if normalized by length of Chain_1)")
            tms = out[k-8:k-1]
        else:
            tms = 'None'
            err.write(str(i+1)+'\t'+str(j+1))
            err.write('\n')
        file.write(str(tms) + '\t')
    file.write('\n')
        
file.close()
err.close()

