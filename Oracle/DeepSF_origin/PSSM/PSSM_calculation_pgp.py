#####################################################################################
# Calculate the PSSM with psiblast and transform it into the correct form for DeepSF.
# 12/09/2019
# Requirement: install blastpgp
# Input: Fasta format sequence
# Output: PSSM (in ascii format and DeepSF format)
#####################################################################################

import sys
import time
import os

start_time = time.time()

SEQ_file = sys.argv[1]
SEQ_name = SEQ_file.split('/')[-1].strip('.fasta')

PSSM_psi_path = sys.argv[2]
PSSM_deepsf_path = sys.argv[3]
PSSM_stat_path = sys.argv[4]
if len(sys.argv) >= 6:
    database = sys.argv[5]
    SEQ_name += '_' + database
else:
    database = '90'

if not PSSM_psi_path.endswith('/'):
    PSSM_psi_path += '/'
if not PSSM_deepsf_path.endswith('/'):
    PSSM_deepsf_path += '/'
if not PSSM_stat_path.endswith('/'):
    PSSM_stat_path += '/'

############################ Calculate the sequence length ##########################

with open(SEQ_file,'r') as seq_file:
    lines = seq_file.readlines()
    seq_len = sum([len(line.strip('\n')) for line in lines[1:]])

############################ Calculate the PSSM with PSI ############################

PSSM_begin_time = time.time()

blastpgp_path = 'blast-2.2.26/bin/blastpgp'

command = blastpgp_path + ' -i ' + SEQ_file 
command += ' -o ' + PSSM_psi_path + SEQ_name + '.blastpgp'
command += ' -j 2 -e 0.001 -h 1e-10 -d nr' + database + '/nr' + database 
command += ' -Q ' + PSSM_psi_path + SEQ_name + '.pssm'
os.system(command)

###################### Transform the PSSM into correct form ##########################

Transform_begin_time = time.time()

pssm_file = open(PSSM_psi_path + SEQ_name + '.pssm','r')
pssm_lines = pssm_file.readlines()
pssm_file.close()

pssm_len = 0
pssm_value_1 = []
#pssm_value_2 = []

for line in pssm_lines[3:]:
    if line == '\n':
        break
    else:
        line = [ch for ch in line.strip('\n').split(' ') if ch != '']
        line_1 = [int(i) for i in line[2:22]]
        #line_2 = [int(i) for i in line[22:42]]
        pssm_value_1 += line_1
        #pssm_value_2 += line_2
        pssm_len += 1

pssm_index_1 = 0
pssm_deepsf_line_1 = ''
for v in pssm_value_1:
    pssm_index_1 += 1
    pssm_deepsf_line_1 += str(pssm_index_1) + ':' + str(v) + ' '
pssm_deepsf_line_1 = pssm_deepsf_line_1.strip(' ') + '\n'

pssm_deepsf_1 = open(PSSM_deepsf_path + SEQ_name + '_pgp.pssm_fea','w')
pssm_deepsf_1.write('>' + SEQ_name + '\n0\t')
pssm_deepsf_1.write(pssm_deepsf_line_1)
pssm_deepsf_1.close()

#pssm_index_2 = 0
#pssm_deepsf_line_2 = ''
#for v in pssm_value_2:
#    pssm_index_2 += 1
#    pssm_deepsf_line_2 += str(pssm_index_2) + ':' + str(v) + ' '
#pssm_deepsf_line_2 = pssm_deepsf_line_2.strip(' ') + '\n'

#pssm_deepsf_2 = open(PSSM_deepsf_path + SEQ_name + '_2_pgp.pssm_fea','w')
#pssm_deepsf_2.write('>' + SEQ_name + '\n0\t')
#pssm_deepsf_2.write(pssm_deepsf_line_2)
#pssm_deepsf_2.close()

End_time = time.time()

############################## Record the statistics ###############################

PSSM_stat_file = open(PSSM_stat_path + SEQ_name + '_pgpstat','w')
PSSM_stat_file.write('Sequence Length: ' + str(seq_len) + '\n')
PSSM_stat_file.write('Running Time: ' + str(End_time - start_time) + '\n')
PSSM_stat_file.write('PSSM Calculation Time: ' + str(Transform_begin_time - PSSM_begin_time) + '\n')
PSSM_stat_file.write('Transformation Time: ' + str(End_time - Transform_begin_time) + '\n')

if pssm_len != seq_len:
    print 'Length Error! seq_len = %d, pssm_len = %d' %(seq_len,pssm_len)
    PSSM_stat_file.write('Length Error! seq_len = ' + str(seq_len) + ', pssm_len = ' + str(pssm_len) + '\n')

PSSM_stat_file.close()



