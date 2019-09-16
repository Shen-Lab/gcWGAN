#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 23:02:46 2018

@author: shaowen1994
"""

import sys
import matplotlib.pyplot as plt
import numpy as np

#path = sys.argv[1]
#path = "seqTest.fa"
path = "astral-scopedom-seqres-gd-sel-gs-bib-100-2.07.fa"

file = open("../Original/"+path,"r")
new_file = open("Test_"+path,"w")
list_row= file.readlines()

num_line = len(list_row)
Sequence = []
Folds = []
Length = []

for i in range(num_line):
    if list_row[i][0] == '>':
        if i != 0:
            Sequence.append(sequence)
            Length.append(len(sequence))
        sequence = ''
        fold = list_row[i].strip("\n").split(' ')[1]
        fold = fold.split('.')
        fold = fold[0]+'.'+fold[1]
        Folds.append(fold)
    else:
        sequence += list_row[i].strip("\n")
        if i == num_line - 1:
            Sequence.append(sequence)
            Length.append(len(sequence))

file.close()
new_file.close()

Length = np.array(Length)
plt.figure(1)
plt.hist(Length,bins = 1000, normed = True)
plt.title("PDF of Sequences")
plt.xlabel("Sequence Length")
plt.xticks(range(0,max(Length),200))
plt.figure(2)
plt.hist(Length,bins = 1000, normed = True, cumulative = True)
plt.title("CDF of Sequences")
plt.xlabel("Sequence Length")
plt.xticks(range(0,max(Length),200))
plt.show()
