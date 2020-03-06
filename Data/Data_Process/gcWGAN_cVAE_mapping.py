import TM_helper
import os
import sys
import DataLoading
import pickle

kind = sys.argv[1]
method = sys.argv[2]
output_file = sys.argv[3]

###################### Load gcWGAN Data #################################

rep_file = open('../Datasets/Origin_SCOPE/represent_file','r')
rep_lines = rep_file.readlines()
rep_file.close()
rep_len = len(rep_lines)

fold_pdb_dic = {}

DATA_DIR = '../Datasets/Final_Data/'
if kind == 'train':
    fold_gc = DataLoading.file_list(DATA_DIR + 'unique_fold_train')
elif kind == 'vali':
    fold_gc = DataLoading.file_list(DATA_DIR + 'fold_val') 
elif kind == 'test':
    fold_gc = DataLoading.file_list(DATA_DIR + 'fold_test')
elif kind == 'nov':
    fold_gc = ['nov']
    fold_pdb_dic['nov'] = ['polb1_tst_106pol_2o02_0073_3.pdb','../Datasets/Final_Data/novel_groundtruth.pdb']
elif kind == 'all':
    fold_gc = DataLoading.file_list(DATA_DIR + 'unique_fold')
    fold_gc.append('nov')
    fold_pdb_dic['nov'] = ['polb1_tst_106pol_2o02_0073_3.pdb','../Datasets/Final_Data/novel_groundtruth.pdb']
  
for i in range(rep_len):
    if rep_lines[i] != '\n':
        line = rep_lines[i].strip('\n').split(' ')
        line_list = []
        for j in line:
            if j != '':
                line_list.append(j)
        pdb_name = line_list[0].upper() 
        fo = line[1].split('.')
        fold = fo[0] + '.' + fo[1]
        fold_pdb_dic[fold] = [pdb_name,'../Datasets/Final_Data/pdbs/' + str(i + 1) + '.pdb']

###################### Load cVAE Data #################################

pdb_DIR = '../Datasets/cVAE_Data/cVAE_pdbs_noTER/'
grammar_dic = {}

flag_dire = False
flag_corr = False

if 'dire' in method:
    flag_dire = True
elif 'corr' in method:
    flag_corr = True
elif 'all' in method:
    flag_dire = True
    flag_corr = True
else:
    print 'Error! Wrong method!'
    quit()

cVAE_pdb_list = [i.strip('.pdb') for i in os.listdir(pdb_DIR) if i.endswith('.pdb')]
if len(set(cVAE_pdb_list)) != len(cVAE_pdb_list):
    print 'Error! Replicated pdb_file!'
    quit()

if flag_dire:   
    ### Load Data from "assign_by_tmscores.txt" ###
    print 'Dire Activated'
    dire_fil = open(pdb_DIR + 'assign_directly.txt','r')
    lines_dire = dire_fil.readlines()
    dire_fil.close()

    for line in lines_dire[1:]:
        line = [i for i in line.strip('\n').split(' ') if i != '']
        pdb = line[0]
        grammar = line[-1]
        grammar_dic[pdb] = [grammar,pdb_DIR + pdb + '.pdb']

if flag_corr:
    ### Load Data from "assign_by_tmscores.txt" ###
    print 'Corr Activated'
    corr_fil = open(pdb_DIR + 'assign_by_tmscores.txt','r')
    lines_corr = corr_fil.readlines()
    corr_fil.close()
    for line in lines_corr[1:]:
        line = [i for i in line.strip('\n').split(' ') if i != '']
        pdb_corr = line[3]
        grammar = line[-1]
        if pdb_corr in grammar_dic.keys():
            if grammar_dic[pdb_corr][0] != grammar:
                print 'Error! Contratict grammar!'
                quit()
        else:
            grammar_dic[pdb_corr] = [grammar,pdb_DIR + pdb_corr + '.pdb']

print 'Size of applied cVAE set:', len(grammar_dic.keys())

if ('all' in method) and (set(cVAE_pdb_list) != set(grammar_dic.keys())):
    print 'Error! Two sets do not match!'
    quit()

###################### Fold Mapping #################################

write_file = open(output_file,'w') 
write_file.write('fold\tpdb_name\tpdb\tcorr_pbd\tbest_tms\tgrammar\n')
write_file.close()

for f_1 in fold_gc:
    print f_1
    pdb_1 = fold_pdb_dic[f_1][1]
    pdb_index = pdb_1.split('/')[-1]
    pdb_name = fold_pdb_dic[f_1][0]
    best_tms = 0
    for pdb_2 in grammar_dic.keys():
        if 'ref' in method:
            tms = TM_helper.TM_score_ref(pdb_1,grammar_dic[pdb_2][1])
        else:
            tms = TM_helper.TM_score(pdb_1,grammar_dic[pdb_2][1])
        if tms > best_tms:
            best_tms = tms
            best_pdb = pdb_2
    gram = grammar_dic[best_pdb][0]
    if 'accum' in method:
        if pdb_name in grammar_dic.keys():
            if grammar_dic[pdb_name][0] != gram:
                print 'Error! Contradict grammar while mapping!'
                print pdb_name,pdb_index
        else:
            grammar_dic[pdb_name + '(' + pdb_index + ')'] = [gram,pdb_1]
    write_file = open(output_file,'a')
    write_file.write(f_1 + '\t' + pdb_name + '\t' + pdb_index + '\t' + best_pdb + '\t' + str(best_tms) + '\t' + gram + '\n')
    write_file.close()
