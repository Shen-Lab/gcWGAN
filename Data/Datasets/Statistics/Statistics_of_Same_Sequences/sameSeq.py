#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 21:18:20 2018

@author: shaowen1994
"""

import sys

#path = sys.argv[1]
path = "all_test.fa"

file = open("../Original/"+path,"r")
new_file = open("Same_Sequence_"+path,"w")
list_row= file.readlines()
num_line = len(list_row)
i = 0
Sequence = []
scanned = []
while(i < num_line):
    if (i in scanned):
        i += 1
    else:
        sequence = ""
        l_1 = 0
        while((i < num_line) and (list_row[i][0] != ">")):
           sequence += list_row[i].strip("\n")
           l_2 += 1
           i += 1
        i += 1
        j = i
        k = 1
        while(j < num_line):
            if j in scanned:
                j += 1
            else:
                sequence_2 = ""
                l_2 = 0
                while((j < num_line) and (list_row[j][0] != ">")):
                     sequence_2 += list_row[j].strip("\n")
                     l_2 += 1
                     j += 1
                j += 1
                if sequence_2 == sequence:
                     for a in range(l_1+3):
                         scanned.append(i-a)   
                     for b in range(l_2+3):
                         scanned.append(i-b)  
                     if k == 1:
                         new_file.write(list_row[i-l_2-2])
                         new_file.write(sequence)
                         new_file.write("\n")
                     k += 1
                     new_file.write(list_row[j-l_2-2])
                     new_file.write(sequence_2)
                     new_file.write('\n')
            if ((j >= num_line) and (k!=1)):
                new_file.write(str(k))
                new_file.write('\n')
                new_file.write('\n')    
file.close()
new_file.close()
