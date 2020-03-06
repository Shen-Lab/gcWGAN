########################################################################################
# Fint out the sequences with high sequence identity to the natral ones and show their 
# alignments.
# 11/11/2019
# Input: fold name
# Output: Record the 5 highest sequence identity and the related sequence alignments
########################################################################################

import sys
import os
import DataLoading
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
import Assessment

matrix = matlist.blosum62

FOLD = sys.argv[1]
fold_name = '_'.join(FOLD.split('.'))

PATH = '../../Results/Diversity_and_Novelty/'
OUT_PATH = PATH + 'Outlier_Check/'
if not os.path.exists(OUT_PATH):
    os.system('mkdir ' + OUT_PATH)
out_file = open(OUT_PATH + 'outlier_check_' + fold_name,'w')
out_file.close()

SELECT_NUM = 5

################################### Load Data ##########################################

def case_columns_to_lists(file_name):
    fil = open(file_name,'r')
    lines = fil.readlines()
    fil.close()
    n = len(lines[0].strip('\n').split('\t'))
    result = []
    for i in range(n):
        result.append([])
    for line in lines:
        line = line.strip('\n').split('\t')
        if len(line) == n:
            for i in range(n):
                result[i].append(line[i])
    return result

def sequence_load(seq_path):
    sequences = Assessment.file_list(seq_path)
    seqs = []
    for s in sequences:
        if s[0]!= '>':
            seqs.append(s)
    return seqs

if FOLD == 'nov':
    nov_cWGAN_s = [float(i) for i in DataLoading.columns_to_lists(PATH + 'cWGAN_Novelty_Successful_' + fold_name)[0][0:-2]]
    nov_cWGAN_r = [float(i) for i in DataLoading.columns_to_lists(PATH + 'cWGAN_Novelty_Random_' + fold_name)[0][0:-2]]
    nov_gcWGAN_s = [float(i) for i in DataLoading.columns_to_lists(PATH + 'gcWGAN_Novelty_Successful_' + fold_name)[0][0:-2]]
    nov_gcWGAN_r = [float(i) for i in DataLoading.columns_to_lists(PATH + 'gcWGAN_Novelty_Random_' + fold_name)[0][0:-2]]
    nov_cVAE = [float(i) for i in DataLoading.columns_to_lists(PATH + 'cVAE_Novelty_' + fold_name)[0][0:-2]]
    nov_cVAE_noX = [float(i) for i in DataLoading.columns_to_lists(PATH + 'cVAE_Novelty_noX_' + fold_name)[0][0:-2]]
else:
    nov_cWGAN_s = [float(i) for i in case_columns_to_lists(PATH + 'cWGAN_Novelty_Successful_' + fold_name)[1]]
    nov_cWGAN_r = [float(i) for i in case_columns_to_lists(PATH + 'cWGAN_Novelty_Random_' + fold_name)[1]]
    nov_gcWGAN_s = [float(i) for i in case_columns_to_lists(PATH + 'gcWGAN_Novelty_Successful_' + fold_name)[1]]
    nov_gcWGAN_r = [float(i) for i in case_columns_to_lists(PATH + 'gcWGAN_Novelty_Random_' + fold_name)[1]]
    nov_cVAE = [float(i) for i in case_columns_to_lists(PATH + 'cVAE_Novelty_' + fold_name)[1]]
    nov_cVAE_noX = [float(i) for i in case_columns_to_lists(PATH + 'cVAE_Novelty_noX_' + fold_name)[1]]

cVAE_seq = sequence_load('cVAE_Samples/cVAE_100_' +  fold_name)
cVAE_seq_noX = sequence_load('cVAE_Samples/cVAE_100_noX_' +  fold_name)
cWGAN_seq_r = sequence_load('../Sequence_Generation/Pipeline_Sample/cWGAN_Fasta_100_random_' +  fold_name)
cWGAN_seq_s = sequence_load('../Sequence_Generation/Pipeline_Sample/cWGAN_Fasta_100_success_' +  fold_name)
gcWGAN_seq_r = sequence_load('../Sequence_Generation/Pipeline_Sample/gcWGAN_Fasta_100_random_' +  fold_name)
gcWGAN_seq_s = sequence_load('../Sequence_Generation/Pipeline_Sample/gcWGAN_Fasta_100_success_' +  fold_name)

if FOLD == 'nov':
    s_n = ''
    nov_fil = open('../../Data/Datasets/Final_Data/nov_sequence','r')
    lines = nov_fil.readlines()
    for line in lines:
        s_n += line.strip('\n')
    nature_seq = [s_n]
else:
    inter_dic = Assessment.Interval_dic('../../Data/Datasets/Final_Data/Interval_1.fa')

    nature_seq = []
    for s in inter_dic[FOLD][1:]:
        nature_seq.append(s.upper())

############################# Record the outliers ######################################

def max_index(data,NUM):
    select_data = sorted(data, reverse = True)[:NUM]
    select_index = [data.index(i) for i in select_data]
    return select_data,select_index

def write_alignment(novelty,seq_gen,seq_nature,NUM,file_w):
    select_iden,select_index = max_index(novelty,NUM)
    for i in range(NUM):
        Iden = select_iden[i]
        Index = select_index[i]
        file_w.write('No. ' + str(i+1) + ': ' + 'Seq_Iden = ' + str(Iden) + '\t' + 'Index = ' + str(Index) + '\n')
        seq_g = seq_gen[Index]
        max_iden = 0
        for seq_n in nature_seq:
            identity = Assessment.Identity(seq_g,seq_n,matrix = matrix)
            if identity > max_iden:
                max_iden = identity
                best_seq = seq_n
        if str(Iden) != str(max_iden):
            print "Error! Identity doesn't match!",Index,Iden,max_iden
        file_w.write(seq_g + '\n')
        file_w.write(best_seq + '\n\n')
        file_w.write('Alignments:\n')
        alignments = pairwise2.align.globaldd(seq_g,best_seq, matrix,-11,-1,-11,-1)
        for j in alignments:
            file_w.write(str(j) + '\n')
        file_w.write('\n')
    return 0

###################################### cVAE ############################################

out_file = open(OUT_PATH + 'outlier_check_' + fold_name,'a')
out_file.write('cVAE:\n')
out_file.write('with X:\n')

write_alignment(nov_cVAE,cVAE_seq,nature_seq,SELECT_NUM,out_file)

out_file.write('without X:\n')

write_alignment(nov_cVAE_noX,cVAE_seq_noX,nature_seq,SELECT_NUM,out_file)

out_file.write('\n')
out_file.close()

##################################### cWGAN ############################################

out_file = open(OUT_PATH + 'outlier_check_' + fold_name,'a')
out_file.write('cWGAN:\n')
out_file.write('Random:\n')

write_alignment(nov_cWGAN_r,cWGAN_seq_r,nature_seq,SELECT_NUM,out_file)

out_file.write('Successful:\n')

write_alignment(nov_cWGAN_s,cWGAN_seq_s,nature_seq,SELECT_NUM,out_file)

out_file.write('\n')
out_file.close()

##################################### gcWGAN ###########################################

out_file = open(OUT_PATH + 'outlier_check_' + fold_name,'a')
out_file.write('gcWGAN:\n')
out_file.write('Random:\n')

write_alignment(nov_gcWGAN_r,gcWGAN_seq_r,nature_seq,SELECT_NUM,out_file)

out_file.write('Successful:\n')

write_alignment(nov_gcWGAN_s,gcWGAN_seq_s,nature_seq,SELECT_NUM,out_file)

out_file.write('\n')
out_file.close()
