#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 10:44:55 2018

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

def matrix_to_file(matrix,path):
    row_num = matrix.shape[0]
    m_file = open(path,'w')
    for i in range(row_num):
        line = [str(m) for m in matrix[i]]
        line = '\t'.join(n for n in line)
        m_file.write(line + '\n')
    m_file.close()
 
for i in range(1,7):
    path = 'TM_scores_' + str(i) + '.fa'
    if i == 1:
        TM_S = readfile(path)
    else:
        TM_S = np.r_[TM_S,readfile(path)]

TM_matrix = 0.5*(TM_S + np.transpose(TM_S))
value, vector = np.linalg.eig(TM_matrix)
matrix_to_file(TM_matrix,'TM_matrix.fa')
matrix_to_file(vector,'Eigenvectors.fa')
s_value = sorted(value,reverse = True)
Value = np.c_[value,s_value]
matrix_to_file(Value,'Eigenvalues.fa')

sum_v = 0
sum_value = []
for i in s_value:
    sum_v += i
    sum_value.append(sum_v)
    
tm_scores = TM_matrix.reshape(-1)

print(s_value[0:20])
plt.figure(1)
plt.hist(s_value,bins = 200,normed = True)
plt.title('Ditribution of Eigenvalues.')
plt.xlabel('Eigenvalue')
plt.ylabel('Distribution Density')
plt.figure(2)
plt.hist(s_value[10:],bins = 200,normed = True)
plt.title('Distribution of the Eigenvalues (except the largest 10 values)')
plt.xlabel('Eigenvalue')
plt.ylabel('Distribution Density')
plt.figure(3)
plt.plot(np.log(s_value))
plt.title('Sorted Eigenvalues')
plt.ylabel('log(eigenvalue)')
plt.figure(4)
plt.plot(sum_value)
plt.title('Sum of Eigenvalues')
plt.figure(5)
plt.hist(tm_scores,bins = 500,normed = True)
plt.title('Ditribution of TM_scores.')
plt.xlabel('TM_score')
plt.ylabel('Distribution Density')
