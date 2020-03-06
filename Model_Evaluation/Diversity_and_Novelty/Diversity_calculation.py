import numpy as np
import Assessment
import sys
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist #SZ add
import os

matrix = matlist.blosum62

fold = sys.argv[1]

if 'nov' in fold:
    fold_name = 'nov'
elif '.' in fold:
    fold_name = fold.split('.')
    fold_name = fold_name[0] + '_' + fold_name[1]
else:
    print 'Error! Wrong fold name!'

model = sys.argv[2]
if model == 'cWGAN' or model == 'gcWGAN':
    kind = sys.argv[3] + '_'
    sample_path = '../Sequence_Generation/Pipeline_Sample/'
    seq_path = sample_path + model + '_Fasta_100_'+ kind +  fold_name
elif model == 'cVAE':   
    if len(sys.argv) >= 4:
        if sys.argv[3] == 'noX':
            kind = sys.argv[3] + '_'
        else:
            kind = ''
    else:
        kind = ''
    sample_path = 'cVAE_Samples/'
    seq_path = sample_path + model + '_100_'+ kind +  fold_name
else:
    print 'Error! Wrong model name!'
    quit()

if kind == 'random_':
    KIND = 'Random_'
elif kind == 'success_':
    KIND = 'Successful_'
elif kind == 'noX_':
    KIND = 'noX_' 
else:
    KIND = ''

if not os.path.exists('../../Results'):
    os.system('mkdir ../../Results')

result_path = '../../Results/Diversity_and_Novelty/'

if not os.path.exists(result_path):
    os.system('mkdir ' + result_path)

iden_fil = open(result_path + model + '_Identity_'+ KIND + fold_name,'w')

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
