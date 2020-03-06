###########################################################################
# Copy all the known pdbs into the target folder with related folds names.
###########################################################################

import sys
import os

path = sys.argv[1]

if not path.endswith('/'):
    path += '/'

with open('../Datasets/Origin_SCOPE/represent_file','r') as f:
    lines = f.readlines()

l = len(lines)

print '%d original pdbs in all.'%l

for i in range(l):
    line = [j for j in lines[i].strip('\n').split(' ') if j != '']
    fold_ch = line[1].split('.')
    fold = fold_ch[0] + '.' + fold_ch[1]
    fold_name = fold_ch[0] + '_' + fold_ch[1]
    print fold,' pdb index:',i+1
    os.system('cp ../Datasets/Final_Data/pdbs/' + str(i+1) + '.pdb ' + path + fold_name + '.pdb')
 
