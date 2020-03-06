import sys
import TM_helper
import DataLoading
import os

object_file = sys.argv[1]
base_path = sys.argv[2]
record_file = sys.argv[3]

if not base_path.endswith('/'):
    base_path += '/'

object_list = DataLoading.columns_to_lists(object_file)[0]

with open(record_file,'w') as f_w:
    f_w.write('fold\tsymmetric\ttm-score\tcVAE_refer\ttm-score\tgcWGAN_refer\ttm-sore\n')

base_list = [base_path + p for p in os.listdir(base_path)]

print '%d gcWGAN folds'%len(object_list)
print '%d cVAE folds'%len(base_list)

gcWGAN_DIR = '../Datasets/Final_Data/pdbs_withName/' 

for f in object_list:
    print f
    f_name = '_'.join(f.split('.'))
    pdb_gcWGAN = gcWGAN_DIR + f_name + '.pdb'
    best_tms_sym = 0 
    best_tms_cVAE = 0
    best_tms_gcWGAN = 0
    for f_cVAE in base_list:
        f_cVAE_ID = f_cVAE.split('/')[-1].strip('.pdb')
        tms_sym = TM_helper.TM_score(pdb_gcWGAN,f_cVAE)
        tms_cVAE = TM_helper.TM_score_ref(pdb_gcWGAN,f_cVAE)
        tms_gcWGAN = TM_helper.TM_score_ref(f_cVAE,pdb_gcWGAN)
        if tms_sym > best_tms_sym:
            best_tms_sym = tms_sym
            best_pdb_sym = f_cVAE_ID
        if tms_cVAE > best_tms_cVAE:
            best_tms_cVAE = tms_cVAE
            best_pdb_cVAE = f_cVAE_ID
        if tms_gcWGAN > best_tms_gcWGAN:
            best_tms_gcWGAN = tms_gcWGAN
            best_pdb_gcWGAN = f_cVAE_ID
    with open(record_file,'a') as f_w:
        f_w.write(f + '\t' + best_pdb_sym + '\t' + str(best_tms_sym) + '\t')
        f_w.write(best_pdb_cVAE + '\t' + str(best_tms_cVAE) + '\t')
        f_w.write(best_pdb_gcWGAN + '\t' + str(best_tms_gcWGAN) + '\n')
