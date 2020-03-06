######################################################################################
# extract the represent sequence with related information
######################################################################################

import sys
import DataLoading

FOLD = sys.argv[1]
PDB = sys.argv[2].lower()
CHAIN = sys.argv[3]
RESI = sys.argv[4]

if RESI == 'None':
    RESI = None
elif RESI[0] == '-':
    RESI = RESI.split('-')
    RESI[0]  = '-' + RESI[0]
else:
    RESI = RESI.split('-')

info = {'pdb_id':PDB,'fold':FOLD,'chain':CHAIN,'resi':RESI}

output_path = sys.argv[5]

print info
 
with open('../Datasets/Origin_SCOPE/astral-scopedom-seqres-gd-sel-gs-bib-100-2.07.fa','r') as ori_seq_file:
    ori_lines = ori_seq_file.readlines()
    l_seq_ori = len(ori_lines)
    i = 0
    while (i < l_seq_ori):
        if '>' in ori_lines[i]:
            if i != 0:
                MATCHED = False
                for info_sco in scope_info:
                    if info == info_sco: 
                        MATCHED = True
                        break
                if MATCHED:
                    final_title = title + '\n'
                    final_seq = seq
                    break

            ############# Read the information of the Scope sequences ##################

            title = ori_lines[i].strip('\n')
            scope_info = []

            ### PDB ID ###
               
            PDB_SCO = title[2:6]

            ### fold ###

            fold_scope = title.split(' ')[1]
            fold_scope = fold_scope.split('.')[0] + '.' + fold_scope.split('.')[1]

            ### chain and resi ###

            chain_resi_scope = title.split(' ')[2]

            if chain_resi_scope[0] == '(' and chain_resi_scope[-1] == ')':
                chain_resi_scope = chain_resi_scope.strip('(').strip(')').split(',')
                  
                for in_s in chain_resi_scope:
                    info_scope_dict = {'pdb_id':PDB_SCO,'fold':fold_scope}
                    info_scope_dict['chain'] = in_s[0]  # chain
                    if ('-' in in_s):
                        resi_scope = in_s.split(':')[-1]
                        if resi_scope[0] == '-':
                            resi_scope = [r for r in resi_scope.split('-') if r != '']
                            resi_scope[0] = '-' + resi_scope[0]
                            info_scope_dict['resi'] = resi_scope
                        else:
                            info_scope_dict['resi'] = [r for r in resi_scope.split('-')]
                    else:
                        info_scope_dict['resi'] = None
                    scope_info.append(info_scope_dict)
                # print scope_info
            else:
                print 'Abnormal title:'
                print title

            ############################################################################

            seq = ''
        else:
            seq += ori_lines[i].strip('\n').upper() + '\n'
        i += 1
    MATCHED = False
    for info_sco in scope_info:
        if info == info_sco:
            MATCHED = True
            break
    if MATCHED:
        final_title = title + '\n'
        final_seq = seq

with open(output_path,'w') as new_seq_file:
    new_seq_file.write(final_title)
    new_seq_file.write(final_seq)
