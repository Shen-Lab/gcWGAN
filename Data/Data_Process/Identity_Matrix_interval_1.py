import sys
import Assessment
import numpy as np
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist #SZ add

matrix = matlist.blosum62

fold_file = sys.argv[1]
output_path = sys.argv[2]

if not output_path.endswith('/'):
    output_path += '/'

DATA_DIR = '../Datasets/Final_Data/'
inter_dic = Assessment.Interval_dic(DATA_DIR + 'Interval_1.fa') 
fold_list = Assessment.file_list(fold_file)

for fold in fold_list:
    fold_name = fold.split('.')[0] + '_' + fold.split('.')[1]
    size = len(inter_dic[fold]) - 1
    print fold,size
    iden_matrix = np.zeros([size,size])
    nature_seq = [i.upper() for i in inter_dic[fold][1:]]
    if len(nature_seq) != size:
        print 'Error! Size Error!'
    for j in range(size):
        seq_1 = nature_seq[j]
        for k in range(size):
            seq_2 = nature_seq[k]
            identity = Assessment.Identity(seq_1,seq_2,matrix = matrix)
            iden_matrix[j,k] = identity
    if True in (iden_matrix != iden_matrix.T):
        print 'Asymmetric!'
        iden_matrix = 0.5*(iden_matrix + iden_matrix.T)
    np.save(output_path + fold_name + '_interval_1',iden_matrix)
   
        
