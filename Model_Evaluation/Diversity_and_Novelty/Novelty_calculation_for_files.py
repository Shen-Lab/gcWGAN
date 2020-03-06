import numpy as np
import Assessment
import sys
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist #SZ add
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os

matrix = matlist.blosum62

seq_path = sys.argv[1]

file_name = seq_path.split('/')[-1]
file_name_list = file_name.split('_')
for i in range(len(file_name_list)):
    if 'nov' in file_name_list[i]:
        fold = 'nov'
        break
    elif file_name_list[i] in ['a','b','c','d','e','f','g']:
        fold = file_name_list[i] + '.' + file_name_list[i+1]
        break

print file_name
print fold

if not os.path.exists('../../Results'):
    os.system('mkdir ../../Results')

result_path = '../../Results/Diversity_and_Novelty/'

if not os.path.exists(result_path):
    os.system('mkdir ' + result_path)

iden_fil = open(result_path + 'Novelty_'+ file_name,'w')

sequences = Assessment.file_list(seq_path)
seqs = []
for s in sequences:
    if s[0]!= '>':
        seqs.append(s)

l = len(seqs)

print l

Identity = []
novelty = []
novelty_max = []

if fold == 'nov':
    s_n = ''
    nov_fil = open('../../Data/Datasets/Final_Data/nov_sequence','r')
    lines = nov_fil.readlines()
    for line in lines:
        s_n += line.strip('\n')
    for s_g in seqs:
        Iden = []
        iden = Assessment.Identity(s_g,s_n,matrix = matrix)
        Identity.append(iden)
        iden_fil.write(str(iden) + '\n')

    iden_fil.write('\n')
    iden_fil.write('all mean: ' + str(np.mean(Identity)) + '\n')
    iden_fil.close()

    print np.mean(Identity)
    
else:
    inter_dic = Assessment.Interval_dic('../../Data/Datasets/Final_Data/Interval_1.fa')

    nature_seq = []
    for s in inter_dic[fold][1:]:
        nature_seq.append(s.upper())

    for s_g in seqs:
        Iden = []
        for s_n in nature_seq:
            iden = Assessment.Identity(s_g,s_n,matrix = matrix)
            Identity.append(iden)
            Iden.append(iden)
        novel = np.mean(Iden)
        novel_max = max(Iden)
        novelty.append(novel)
        novelty_max.append(novel_max)
        iden_fil.write(str(novel) + '\t' + str(novel_max) + '\n')
    
    iden_fil.write('\n')
    iden_fil.write('max mean: ' + str(np.mean(novelty_max)) + '\n')
    iden_fil.write('all mean: ' + str(np.mean(Identity)) + '\n')
    iden_fil.close()

    print np.mean(novelty_max)
    print np.mean(Identity)
