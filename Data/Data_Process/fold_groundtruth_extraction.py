####################################################################
# Extract the groundtruth pdb of the input fold to a certain path.
####################################################################

import sys
import os

FOLD = sys.argv[1]
path = sys.argv[2]

if not path.endswith('/'):
    path += '/'

if 'nov' in FOLD.lower():
    fold_name = 'novel'
    os.system('cp  ../Datasets/Novel_Fold/polb1_tst_106pol_2o02_0073_3.pdb ' + path + fold_name + '_groundtruth.pdb')
elif '.' in FOLD:
    with open('../Datasets/Origin_SCOPE/represent_file','r') as f:
        lines = f.readlines()
    fold_name = '_'.join(FOLD.split('.'))

    l = len(lines)

    print '%d original pdbs in all.'%l

    for i in range(l):
        line = [j for j in lines[i].strip('\n').split(' ') if j != '']
        fold_ch = line[1].split('.')
        fold = fold_ch[0] + '.' + fold_ch[1]
        if fold == FOLD:
            print 'pdb index:',i+1
            os.system('cp ../Datasets/Final_Data/pdbs/' + str(i+1) + '.pdb ' + path + fold_name + '_groundtruth.pdb')
            break
 
