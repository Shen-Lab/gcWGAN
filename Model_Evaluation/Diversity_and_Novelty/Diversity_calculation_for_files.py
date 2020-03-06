import numpy as np
import Assessment
import sys
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist #SZ add
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

iden_fil = open(result_path + 'Identity_' + file_name,'w')

sequences = Assessment.file_list(seq_path)
seqs = []
for s in sequences:
    if s[0]!= '>':
        seqs.append(s)

l = len(seqs)

print l

Identity = []

for i in range(l):
    for j in range(i+1,l):
        iden = Assessment.Identity(seqs[i],seqs[j],matrix = matrix)
        Identity.append(iden)
        iden_fil.write(str(iden) + '\n')

iden_fil.write('\n')
iden_fil.write(str(np.mean(Identity)) + '\n')
iden_fil.close()

print np.mean(Identity)
