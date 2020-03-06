######################################################################################
# extract the represent sequence for the input fold or pdb ID
######################################################################################

import sys
import DataLoading

INPUT = sys.argv[1]
output_path = sys.argv[2]
if not output_path.endswith('/'):
    output_path += '/'

if 'nov' in INPUT.lower():
    print 'novel'
    with open('../Datasets/Final_Data/nov_sequence','r') as ori_seq_file:
        lines = ori_seq_file.readlines()
    with open(output_path + 'novel_seq.fasta','w') as new_seq_file:
        new_seq_file.write('>novel_sequence\n')
        for line in lines:
            new_seq_file.write(line)

elif '.' in INPUT: 
    fold_name = '_'.join(INPUT.split('.'))
    with open('../Datasets/Origin_SCOPE/represent_file','r') as f:
        rep_lines = f.readlines()
        l = len(rep_lines)
    for i in range(l):
        line = [j for j in rep_lines[i].strip('\n').split(' ') if j != '']
        fold_ch = line[1].split('.')
        fold = fold_ch[0] + '_' + fold_ch[1]
        if fold == fold_name:
            repre_info = DataLoading.read_repredata(rep_lines[i])
            PDB_ID = repre_info[0]['pdb_id']
            break
    print fold_name
    print PDB_ID
    print repre_info
        
    with open('../Datasets/Origin_SCOPE/astral-scopedom-seqres-gd-sel-gs-bib-100-2.07.fa','r') as ori_seq_file:
        ori_lines = ori_seq_file.readlines()
        l_seq_ori = len(ori_lines)
        i = 0
        while (i < l_seq_ori):
            if '>' in ori_lines[i]:
                if i != 0:
                    MATCHED = False
                    for info in repre_info:
                        for info_sco in scope_info:
                            if info == info_sco: 
                                MATCHED = True
                                break
                        if MATCHED:
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
        for info in repre_info:
            for info_sco in scope_info:
                if info == info_sco:
                    MATCHED = True
                    break
            if MATCHED:
                break
        if MATCHED:
            final_title = title + '\n'
            final_seq = seq

    with open(output_path + fold_name + '_seq.fasta','w') as new_seq_file:
        new_seq_file.write(final_title)
        new_seq_file.write(final_seq)

elif (len(INPUT) == 4) and (INPUT[0] in ['1','2','3','4','5','6','7','8','9','0']):
    PDB_ID = INPUT
    print PDB_ID
    
    with open('../Datasets/Origin_SCOPE/astral-scopedom-seqres-gd-sel-gs-bib-100-2.07.fa','r') as ori_seq_file:
        ori_lines = ori_seq_file.readlines()
        l_seq_ori = len(ori_lines)
        i = 0
        while (i < l_seq_ori):
            if '>' in ori_lines[i]:
                if i != 0:
                    if PDB_ID.lower() == title[2:6]:
                        final_title = title
                        final_seq = seq
                        break
                title = ori_lines[i]
                seq = ''
            else:
                seq += ori_lines[i].strip('\n').upper() + '\n'
            i += 1
        if PDB_ID.lower() == title[2:6]:
            final_title = title
            final_seq = seq

    with open(output_path + PDB_ID + '_seq.fasta','w') as new_seq_file:
        new_seq_file.write(final_title)
        new_seq_file.write(final_seq)

