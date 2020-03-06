import os

DIR = '../Datasets/cVAE_Data/cVAE_pdbs/'

pdb_list = [p for p in os.listdir(DIR) if p.endswith('.pdb')]

for pdb in pdb_list:
    fil = open(DIR + pdb,'r')
    lines = fil.readlines()
    fil.close()
    flag = False
    for line in lines:
        if 'TER' in line:
            TER_index = lines.index(line) + 1
            flag = True
        if flag and 'ATOM' in line:
            print 'TER!',pdb,TER_index,lines.index(line) + 1
            print ''
            break
     
