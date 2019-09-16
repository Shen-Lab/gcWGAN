import numpy as np
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist 
import Assessment
import sys 
matrix = matlist.blosum62

DIR = 'data_1/'

batch_num = int(sys.argv[1])
batch_index = int(sys.argv[2])
dic = Assessment.Train_dic(DIR + 'fold_train', DIR + 'seq_train')
length = len(dic.keys())
batch_size = float(length)/float(batch_num)
Batch = [int(batch_size)]*(batch_num - 1)
Batch.append(length - int(batch_size)*(batch_num - 1))

if sum(Batch) != length:
    print 'Batch Error! Batch sum = ',sum(Batch),',length = ',length
else:
    print 'Load Data successfully!'

    id_file = open(DIR + 'Train_Identity_Matrix_'+str(batch_index),'w')
    id_file.close()

    for f in dic.keys()[sum(Batch[0:batch_index]):sum(Batch[0:batch_index+1])]:
        l = len(dic[f])
        id_file = open(DIR + 'Train_Identity_Matrix_'+str(batch_index),'a')
        id_file.write(f + '\t' + str(l) + '\n')
        i_matrix = np.zeros([l,l])
        for i in xrange(l):
            for j in xrange(l):
                iden = Assessment.Identity(dic[f][i],dic[f][j],matrix)
                i_matrix[i][j] = iden
                id_file.write(str(iden) + '\t')
            id_file.write('\n')
        if np.allclose(i_matrix,i_matrix.T):
            print 'Fold '+ f + ' solved.'
            id_file.write('Fold '+ f + ' solved.\n')
        else:
            print 'Fold '+ f + 'asymmetric.'
            id_file.write('Fold '+ f + ' asymmetric.\n')
        id_file.write('\n')
        id_file.close()
