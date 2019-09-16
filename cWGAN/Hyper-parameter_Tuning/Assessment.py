#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:11:19 2018

@author: shaowen1994
"""

import numpy as np
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist #SZ add

matrix = matlist.blosum62

def Interval_dic(path):
    """
    Load Interval file and return a dictionary.
    Keys are folds name.
    Values are lists, where first element is the fold number and others are sequences.
    """
    fil = open(path)
    lines = fil.readlines()
    dic = {}
    for i in lines:
        line = i.strip('\n')
        if line[1] == '.' :
            line = line.split('\t')
            fold = line[0]
            dic[fold] = [int(line[1])]
        else:
            dic[fold].append(line)
    fil.close()
    return dic

def Train_dic(path_1,path_2):
    folds = open(path_1,'r')
    seqs = open(path_2,'r')
    lines_f = folds.readlines()
    lines_s = seqs.readlines()
    if len(lines_f) != len(lines_s):
        print 'Input unrelated files!'
        return None
    else:
        f_s_dic = {}
        for i in xrange(len(lines_f)):
            f = lines_f[i].strip('\n')
            s = lines_s[i].strip('\n')
            if f in f_s_dic.keys():
                f_s_dic[f].append(s)
            else:
                f_s_dic[f] = [s]
    return f_s_dic

def file_dic(seq_path,fold_path):
    """
    Load test file and return a dictionary.
    Keys are folds name.
    Values are lists, where first element is the fold number and others are sequences.
    """
    s_file = open(seq_path)
    f_file = open(fold_path)
    s_lines = s_file.readlines()
    f_lines = f_file.readlines()
    l_s = len(s_lines)
    l_f = len(f_lines)
    if l_s != l_f:
        print "Input wrong file"
        return 0
    dic = {}
    for i in xrange(l_s):
        fold = f_lines[i].strip('\n')
        seq = s_lines[i].strip('\n')
        if fold in dic.keys():
            dic[fold].append(seq)
        else:
            dic[fold] = [seq]
    s_file.close()
    f_file.close()
    return dic

def file_list(path):
    f = open(path,'r')
    lines = f.readlines()
    result = []
    for i in lines:
       line = i.strip('\n')
       result.append(line)
    return result

def representative_dic(path,dic):
    d_c = {}
    fil = open(path,'r')
    lines = fil.readlines()
    l = len(lines)
    i = 0
    while(i < l):
        if  lines[i][1] == '.':
            fold = lines[i].strip('\n')
            index = lines[i+1].strip('\n').split(' ')[:-1]
            r_seq = [dic[fold][int(j)] for j in index] 
            if not (fold[0] in d_c):
                d_c[fold[0]] = {fold:r_seq}
            else:
                d_c[fold[0]][fold] = r_seq
            i += 3
        else:
            i += 1     
    fil.close()
    if len(d_c.keys()) == 7 and l == len(dic.keys())*3:
        return d_c
    else:
        print 'Error! Wrong folds number!'
        return 0
    
def delete_padding(x):
    """
    x is a sequence.
    Choosw the longest continuous sequence between paddings.
    """
    s = x.split('!')
    l = 0
    seq = ''
    for i in s:
        if len(i) > l:
            l = len(i)
            seq = i
    return seq
 
def alignment_score(x,y,matrix):
    """
    Return 3 values:
    sequence identity
    coverage
    maximum normalized score
    """
    l_x = len(x)
    l_y = len(y)
    l  = min(l_x,l_y)
    X = x.upper()
    Y = y.upper()
    alignments = pairwise2.align.globaldd(X,Y, matrix,-11,-1,-11,-1)  # Consistent with Blast P grobal alignment
    len_min = 10000
    coverage = 0
    max_same = 0
    for i in alignments:
        if i[-1] < len_min:
            len_min = i[-1]
        same = 0
        for j in xrange(i[-1]):
            if i[0][j] == i[1][j] and i[0][j] != '-':
                same += 1
        if float(same)/float(i[-1]) > coverage:
            coverage = float(same)/float(i[-1]) 
        if same > max_same:
            max_same = same
    identity = float(max_same)/float(l)
    n_score = float(alignments[0][-3])/float(len_min)
    return identity,coverage,n_score

def Identity(x,y,matrix = matrix):
    l_x = len(x)
    l_y = len(y)
    X = x.upper()
    Y = y.upper()
    alignments = pairwise2.align.globaldd(X,Y, matrix,-11,-1,-11,-1)   # Consistent with Blast P grobal alignment
    max_same = 0
    for i in alignments:
        max_iden = 0
        same = 0
        for j in xrange(i[-1]):
            if i[0][j] == i[1][j] and i[0][j] != '-':
                same += 1
        iden = float(same)/float(i[-1])
        if iden > max_iden:
            max_iden = iden
    return max_iden

def Gap_excluded_Identity(x,y,matrix = matrix):
    l_x = len(x)
    l_y = len(y)
    l = min(l_x,l_y)
    X = x.upper()
    Y = y.upper()
    alignments = pairwise2.align.globaldd(X,Y, matrix,-11,-1,-11,-1)   # Consistent with Blast P grobal alignment
    max_same = 0
    for i in alignments:
        same = 0
        for j in xrange(i[-1]):
            if i[0][j] == i[1][j] and i[0][j] != '-':
                same += 1
        if same > max_same:
            max_same = same
    identity = float(max_same)/float(l)
    return identity

def output_score(keys,samples,dic_all,matrix,padding = 1):
    """
    Input several samples, the dictionary for the folds and the Blosum matrix.
    Output the average identity, coverage and best alignment score.
    """
    l = len(keys)
    if l != len(samples):
        print "Error! Input wrong arguments!"
        return 0,0,0
    if padding == 1:
        samples = [delete_padding(''.join(s)) for s in samples]
    else:
        samples = [''.join(s) for s in samples]
    #for i in xrange(l):
        #for s_real in dic_all[keys[i]]

def average_alignment(samples_f,rep_s,matrix):
    Iden = []
    Cove = []
    N_ali = []
    for s_1 in samples_f:
        i_max = 0
        c_max = 0
        n_max = 0
        for s_2 in rep_s:
            i,c,n = alignment_score(s_1,s_2,matrix)
            if i > i_max:
                i_max = i
            if c > c_max:
                c_max = c
            if n > n_max:
                n_max = n
        Iden.append(i_max)
        Cove.append(c_max)
        N_ali.append(n_max)
    return np.mean(Iden),np.mean(Cove),np.mean(N_ali)       

def average_score(keys_all,samples,unique_train,unique_new,all_dic,matrix):
    """
    Return the  average scores of all the folds respectively for training samples and test samples.
    """
    l = len(keys_all)
    l_train = len(unique_train)
    l_new = len(unique_new)
    if l != len(samples) or l != l_train + l_new:
        print "Error! Input wrong arguments!"
        return 0
    train_score = 0
    train_identity = 0
    train_coverage = 0
    new_score = 0
    new_identity = 0
    new_coverage = 0
    for i in xrange(l):
        s_1 = delete_padding(''.join(samples[i]))
        best_score = 0
        best_coverage = 0
        best_identity = 0
        for s in all_dic[keys_all[i]][1:]:
            identity,coverage,score  = alignment_score(s_1,s,matrix)
            if score > best_score:
                best_score = score
            if identity > best_identity:
                best_identity = identity
            if coverage > best_coverage:
                best_coverage = coverage        
        if (keys_all[i] in unique_train):   
            train_score += best_score
            train_identity += best_identity
            train_coverage += best_coverage
        else:
            new_score += best_score
            new_identity += best_identity
            new_coverage += best_coverage
    return float(train_identity)/float(l_train),float(train_coverage)/float(l_train),float(train_score)/float(l_train),float(new_identity)/float(l_new),float(new_coverage)/float(l_new),float(new_score)/float(l_new)

def SimilarityMeasure(generated_seq,original_seq):
    """
    Input a list of generated sequences and a list of original sequences. 
    Output the related identity, coverge and normalized alignment score.
    """
    I = []
    C = []
    N = []
    for s_g in generated_seq:
        i_max = 0
        c_max = 0
        n_max = 0
        for s_o in original_seq:
            i,c,n = alignment_score(s_g,s_o,matrix) 
            if i > i_max:
                i_max = i
            if c > c_max:
                c_max = c
            if n > n_max:
                n_max = n 
        I.append(i_max)
        C.append(c_max)
        N.append(n_max)
    return np.mean(I),np.mean(C),np.mean(N)  

def Fasta(seq):
    """
    Transfer a sequence into Fasta format.
    """   
    result = seq.strip('!')
    if '!' in result:
        print 'Invalid Sequence'
        return None
    else:
        result = result.upper()
        return '>q/n' + result

             
