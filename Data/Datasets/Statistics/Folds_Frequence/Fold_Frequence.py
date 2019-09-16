#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 23:02:46 2018

@author: shaowen1994
"""

import sys
import matplotlib.pyplot as plt

#path = sys.argv[1]
path = "astral-scopedom-seqres-gd-sel-gs-bib-100-2.07.fa"

file = open("../Original/"+path,"r")
list_row= file.readlines()

num_line = len(list_row)
Sequence = []
Folds = []

for i in range(num_line):
    if list_row[i][0] == '>':
        if i != 0:
            Sequence.append(sequence)
        sequence = ''
        fold = list_row[i].strip("\n").split(' ')[1]
        fold = fold.split('.')
        fold = fold[0]+'.'+fold[1]
        Folds.append(fold)
    else:
        sequence += list_row[i].strip("\n")
        if i == num_line - 1:
            Sequence.append(sequence)
file.close()

num_folds = len(Folds)

d_fold = {}
for i in range(num_folds):
    if Folds[i] in list(d_fold.keys()):
        d_fold[Folds[i]] += 1
    else:
        d_fold[Folds[i]] = 1

k_500 = 0
k_200 = 0
k_100 = 0     
k_50 = 0
k_20 = 0 
k_10 = 0
k_5 = 0 
l_sorted = sorted(d_fold.items(),key = lambda x:x[1],reverse = True)
kind_fold = len(l_sorted)
d_sorted = {}
for i in range(kind_fold):
    d_sorted[l_sorted[i][0]] = l_sorted[i][1]
    if l_sorted[i][1] <= 500 and k_500 == 0:
        print(l_sorted[i][1],":",i," ",l_sorted[i][0])
        k_500 += 1
    elif l_sorted[i][1] <= 200 and k_200 == 0:
        print(l_sorted[i][1],":",i," ",l_sorted[i][0])
        k_200 += 1
    elif l_sorted[i][1] <= 100 and k_100 == 0:
        print(l_sorted[i][1],":",i," ",l_sorted[i][0])
        k_100 += 1
    elif l_sorted[i][1] <= 50 and k_50 == 0:
        print(l_sorted[i][1],":",i," ",l_sorted[i][0])
        k_50 += 1
    elif l_sorted[i][1] <= 20 and k_20 == 0:
        print(l_sorted[i][1],":",i," ",l_sorted[i][0])
        k_20 += 1
    elif l_sorted[i][1] <= 10 and k_10 == 0:
        print(l_sorted[i][1],":",i," ",l_sorted[i][0])
        k_10 += 1
    elif l_sorted[i][1] <= 5 and k_5 == 0:
        print(l_sorted[i][1],":",i," ",l_sorted[i][0])
        k_5 += 1
        
    
plt.subplot(211)
plt.plot(d_sorted.values())
plt.subplot(212)
plt.plot(list(d_sorted.values())[0:40])
plt.show()


