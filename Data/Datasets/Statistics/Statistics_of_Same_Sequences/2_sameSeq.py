#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 16:06:06 2018

@author: shaowen1994
"""
import sys

path = sys.argv[1]
#path = "seqTest.fa"

file = open("../Original/"+path,"r")
new_file = open("Same_Sequence_"+path,"w")
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

scanned = []         
num_fold = len(Folds)
for i in range(num_fold):
    #print(i)
    if not (i in scanned):
        scanned.append(i)
        k = 1
        for j in range(i+1,num_fold):
            if Sequence[j] == Sequence[i]:
               scanned.append(j)
               if Folds[i] != Folds[j]:
                  # time += 1
                   #print(time)
                   if k == 1:
                      new_file.write("#"+str(i))
                      new_file.write('\n')
                      new_file.write(Folds[i])
                      new_file.write('\n')
                      new_file.write(Sequence[i])
                      new_file.write('\n')
                   k += 1
                   new_file.write("#"+str(j))
                   new_file.write('\n')
                   new_file.write(Folds[j])
                   new_file.write('\n')
                   new_file.write(Sequence[j])
                   new_file.write('\n')
        if (k > 1):
            new_file.write("Appearing Times: " + str(k)+ '\n')
            new_file.write('\n') 

file.close()
new_file.close()
