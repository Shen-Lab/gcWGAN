#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 18:51:43 2018

@author: shaowen1994
"""

import numpy as np
import matplotlib.pyplot as plt

def readfile(path):
    file = open(path,'r')
    data = file.readlines()
    length = len(data)
    for i in range(length):
        line = data[i].strip('\n').split('\t')[0:-1]
        line = [float(x) for x in line]
        line = np.array([line])
        if i == 0:
            result = line
        else:
            result = np.r_[result,line]
    file.close()
    return result

for i in range(1,7):
    path = 'TM_scores_' + str(i) + '.fa'
    if i == 1:
        TM_S = readfile(path)
    else:
        TM_S = np.r_[TM_S,readfile(path)]

TM_matrix = 0.5*(TM_S + np.transpose(TM_S))
tm_scores = TM_matrix.reshape(-1)
s_max = max(tm_scores)
D_pre = s_max - TM_matrix

file = open("Invalid_Distance.fa","w")
n = 0
for i in range(N):
    for j in range(i,N):
        for m in range(j,N):
            n += 1
            if (D_pre[i][j] + D_pre[j][m] < D_pre[i][m]):
                print(i,j,m)
                file.write(str(i)+'\t'+str(j)+'\t'+str(m)+'\t'+str(D_pre[i][j])+'\t'+str(D_pre[j][m])+'\t'+str(D_pre[i][m])+'\n')
            elif (D_pre[i][m] + D_pre[m][j] < D_pre[i][j]):
                print(i,m,j)
                file.write(str(i)+'\t'+str(m)+'\t'+str(j)+'\t'+str(D_pre[i][m])+'\t'+str(D_pre[m][j])+'\t'+str(D_pre[i][j])+'\n')
            elif (D_pre[j][i] + D_pre[i][m] < D_pre[j][m]):
                print(j,i,m)
                file.write(str(j)+'\t'+str(i)+'\t'+str(m)+'\t'+str(D_pre[i][j])+'\t'+str(D_pre[m][i])+'\t'+str(D_pre[j][m])+'\n')
            else:
                n -= 1
file.write(str(n))
file.close()
print(n)