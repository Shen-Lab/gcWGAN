#####################################################################################
# Get the AA, SS and SA features with SCRATCH and transform it into the correct form for DeepSF.
# 11/08/2019
# Requirement: module load BLAST+/2.7.1-intel-2017b-Python-2.7.14
# Input: Fasta format sequence
# Output: aa, ss and sa features (in SCRATCH format and DeepSF format)
#####################################################################################

import sys
import time
import os

start_time = time.time()

SEQ_file = sys.argv[1]
SCRATCH_output_name = SEQ_file.split('/')[-1].strip('.fatsa')

aa_ss_sa_scratch_path = sys.argv[2]
aa_ss_sa_deepsf_path = sys.argv[3]
aa_ss_sa_stat_path = sys.argv[4]

if not aa_ss_sa_scratch_path.endswith('/'):
    aa_ss_sa_scratch_path += '/'
if not aa_ss_sa_deepsf_path.endswith('/'):
    aa_ss_sa_deepsf_path += '/'
if not aa_ss_sa_stat_path.endswith('/'):
    aa_ss_sa_stat_path += '/'

#################### Calculate the features with SCRATCH ############################

SCRATCH_begin_time = time.time()

command = '../DeepSF/software/SCRATCH-1D_1.1/bin/run_SCRATCH-1D_predictors.sh ' + SEQ_file 
command += ' ' + aa_ss_sa_scratch_path + SCRATCH_output_name
os.system(command)

################# Transform the features into correct form ##########################

Transform_begin_time = time.time()

seq_dic = {}

################################# AA ################################ 
with open(SEQ_file,'r') as aa_file:
    aa_lines = aa_file.readlines()

SEQ_num = 0
for line in aa_lines:
    if '>' in line:
        if SEQ_num != 0:
            seq_len = len(seq)
            seq_dic[SEQ_name] = [seq,seq_len]
        SEQ_num += 1
        SEQ_name = line.split(' ')[0][1:]
        seq = ''
    else:
        seq += line.strip('\n')

seq_len = len(seq)
seq_dic[SEQ_name] = [seq,seq_len]
################################# SS ################################
with open(aa_ss_sa_scratch_path + SCRATCH_output_name + '.ss','r') as ss_file:
    ss_lines = ss_file.readlines()

SS_num = 0
for line in ss_lines:
    if '>' in line:
        if SS_num != 0:
            seq_dic[SEQ_name].append(ss)
            if seq_dic[SEQ_name][1] != len(ss):
                print 'Error! Length of AA and SS so not match!'
        SS_num += 1
        SEQ_name = line.split(' ')[0][1:]
        ss = ''
    else:
        ss += line.strip('\n')

seq_dic[SEQ_name].append(ss)
if seq_dic[SEQ_name][1] != len(ss):
    print 'Error! Length of AA and SS so not match!'
################################# SA ################################
with open(aa_ss_sa_scratch_path + SCRATCH_output_name + '.acc','r') as sa_file:
    sa_lines = sa_file.readlines()

SA_num = 0
for line in sa_lines:
    if '>' in line:
        if SA_num != 0:
            seq_dic[SEQ_name].append(sa)
            if seq_dic[SEQ_name][1] != len(sa):
                print 'Error! Length of AA and SA so not match!'
        SA_num += 1
        SEQ_name = line.split(' ')[0][1:]
        sa = ''
    else:
        sa += line.strip('\n')

seq_dic[SEQ_name].append(sa)
if seq_dic[SEQ_name][1] != len(sa):
    print 'Error! Length of AA and SA so not match!'
############################ Statisic ##############################
if (SA_num == SS_num) and (SEQ_num == SS_num):
    print '%d protein sequences detected:'%SEQ_num
    for s in seq_dic.keys():
        print s,seq_dic[s][1]
else:
    print 'Sequence number does not match!'
##################### Record as DeepSF form ########################
def one_hot(seq,dic):
    wide = len(dic.keys())
    result = []
    for ch in seq:
        row = [0]*wide
        row[dic[ch]] = 1
        result.append(row)
    return result

AA_dict = {'A':0,'R':1,'N':2,'D':3,'C':4,'Q':5,'E':6,'G':7,'H':8,'I':9,'L':10,'K':11,'M':12,'F':13,'P':14,'S':15,'T':16,'W':17,'Y':18,'V':19}
SS_dict = {'C':0,'E':1,'H':2}
SA_dict = {'e':0,'-':1}

for seq_name in seq_dic.keys():
    feature = '0\t'
    index = 0
    seq_len = seq_dic[seq_name][1]
    aa_list = one_hot(seq_dic[seq_name][0],AA_dict)
    ss_list = one_hot(seq_dic[seq_name][2],SS_dict)
    sa_list = one_hot(seq_dic[seq_name][3],SA_dict)
    for i in range(seq_len):
        aa = aa_list[i]
        ss = ss_list[i]
        sa = sa_list[i]
        for j in aa:
            index += 1
            feature += str(index) + ':' + str(j) + ' '
        for j in ss:
            index += 1
            feature += str(index) + ':' + str(j) + ' '
        for j in sa:
            index += 1
            feature += str(index) + ':' + str(j) + ' '
        
    feature_deepsf = open(aa_ss_sa_deepsf_path + seq_name + '.fea_aa_ss_sa','w')
    feature_deepsf.write('>' + seq_name + '\n')
    feature_deepsf.write(feature.strip(' ') + '\n')
    feature_deepsf.close()

End_time = time.time()

############################## Record the statistics ###############################

aa_ss_sa_stat_file = open(aa_ss_sa_stat_path + SCRATCH_output_name + '_stat','w')
if (SA_num == SS_num) and (SEQ_num == SS_num):
    aa_ss_sa_stat_file.write(str(SEQ_num) +' protein sequences detected:\n')
    for s in seq_dic.keys():
        aa_ss_sa_stat_file.write(s + ' length: ' + str(seq_dic[s][1]))
else:
    aa_ss_sa_stat_file.write('Sequence number does not match!')

aa_ss_sa_stat_file.write('\n')
aa_ss_sa_stat_file.write('Running Time: ' + str(End_time - start_time) + '\n')
aa_ss_sa_stat_file.write('SCRATCH Time: ' + str(Transform_begin_time - SCRATCH_begin_time) + '\n')
aa_ss_sa_stat_file.write('Transformation Time: ' + str(End_time - Transform_begin_time) + '\n')

aa_ss_sa_stat_file.close()



