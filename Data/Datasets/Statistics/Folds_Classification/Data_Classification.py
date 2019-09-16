#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 21:23:21 2018

@author: shaowen1994
"""

def dic_file(D,file_name):
    file_name += '.fa'
    wfile = open(file_name,'w')
    for f in D:
        wfile.write(f + '\t' + str(D[f][0]) + '\n')
        for s in D[f][1:]:
            wfile.write(s + '\n')
    wfile.close()

file = open("astral-scopedom-seqres-gd-sel-gs-bib-100-2.07.fa")
data = file.readlines()
l = len(data)

All = {}
Easy = {}
Median = {}
Difficult = {}

i = 0
while (i < l):
    line = data[i]
    i += 1
    if line[0] != '>':
        print("Error!")
    fold = line.strip('\n').split(' ')
    fold = fold[1].split('.')
    fold = fold[0] + '.' + fold[1]
    seq =''
    while(data[i][0] != '>'):
        seq += data[i].strip('\n')
        i += 1
        if i >= l:
            break
    if fold in All.keys():
        All[fold][0] += 1
        All[fold].append(seq)
    else:
        All[fold] = [1,seq]
e = 0
m = 0
d = 0
for f in All:
    if All[f][0] <= 5:
        Difficult[f] = All[f]
        d += 1
    elif All[f][0] > 50:
        Easy[f] = All[f]
        e += 1
    else:
        m += 1
        Median[f] = All[f]

dic_file(All,'All')
dic_file(Easy,'Easy')
dic_file(Median,'Median')
dic_file(Difficult,'Difficult')
print(e,m,d)