################################################################################################
# Load Identity matrix and apply DBSCAN algorithm to select the representative sequences.
# Input: path of the matrices
# Output: a file that contain the indexes of the representative seqeunces
################################################################################################
import numpy as np
from sklearn.cluster import DBSCAN
import sys
import os
import DataLoading

mat_path = sys.argv[1]
out_path = sys.argv[2]
if not mat_path.endswith('/'):
    mat_path += '/'

DATA_DIR = '../Datasets/Final_Data/'
if len(sys.argv) >= 4:
    flag = False    
    set_kind = sys.argv[3]
    if set_kind == 'train':
        fold_list = DataLoading.file_list(DATA_DIR + 'unique_fold_train')
    elif set_kind == 'vali':
        fold_list = DataLoading.file_list(DATA_DIR + 'fold_val')
    elif set_kind == 'test':
        fold_list = DataLoading.file_list(DATA_DIR + 'fold_test')
    else:
        print 'Error! No set named %s'%set_kind
else:
    flag = True
    fold_list = []

mat_list = [i for i in os.listdir(mat_path) if i.endswith('.npy')]

fold={}

s=0
s1=0
for mat in mat_list:
    fname = mat.strip('.npy').split('_')
    fname = fname[0] + '.' + fname[1]
    if fname in fold_list or flag:
        ma = np.load(mat_path + mat)
        ma1 = 1. - ma
        clustering = DBSCAN(metric='precomputed', eps=0.7, min_samples=1).fit(ma1)
        re=clustering.labels_
        index=[]				
        score=[]

        for j in range(len(ma1)):
            score.append(np.sum(ma1[j][np.where(re==re[j])[0]]))

        for k in range(np.max(re)+1):
            tt=np.where(re==k)[0][0]
	    for j in np.where(re==k)[0]:
	        if(score[j]<score[tt]):
		    tt=j

	    index.append(tt)

        fold[fname]=index

        s+=len(np.where(re==-1)[0]) + np.max(re)+1
        s1+=len(index)

        print re

print s, s1

with open(out_path, "w") as f:
    for i in fold:
        f.write("%s\n" %(i))
	for j in range(len(fold[i])):
	    f.write("%d " %(fold[i][j]))
	f.write("\n\n")





