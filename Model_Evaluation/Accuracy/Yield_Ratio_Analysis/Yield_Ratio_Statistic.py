import os, sys
sys.path.append(os.getcwd())

import time

import numpy as np
import tensorflow as tf
import data_helpers #SZ change
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import tflib.plot
import DataLoading #SZ add
import matplotlib.pyplot as plt #SZ add
import os

from keras.models import model_from_json
from keras.engine.topology import Layer
import theano.tensor as T
from keras import backend as K
from keras.constraints import maxnorm

from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Merge, Convolution1D
from keras.layers.normalization import BatchNormalization

PATH = sys.argv[1]
EPOCH = sys.argv[2]

train_flag = ('train' in sys.argv)
vali_flag = ('vali' in sys.argv)
test_flag = ('test' in sys.argv)

if not (train_flag or vali_flag or test_flag):
    print 'Error! No dataset input!'
    quit()

SEQ_LEN = 160 # Sequence length in characters
MAX_N_EXAMPLES = 50000

if 'gcWGAN' in PATH:
    result_path = '../../../Results/Accuracy/Yield_Ratio_Result/gcWGAN/'
elif 'cWGAN' in PATH:
    result_path = '../../../Results/Accuracy/Yield_Ratio_Result/cWGAN/'
else:
    print 'Path Error!'
    quit()

if PATH[-1] == '/':
    PATH = PATH.strip('/')

path_split = PATH.split('_')
NAME = ''
flag = 0

for j in path_split:
    if not ('heck' in j or 'o' in j or 'HECK' in j or 'O' in j):
        NAME += j + '_'
        if '.' in j and flag == 0:
            flag = 1
            p_index = path_split.index(j)
            print j
            print p_index

NAME += EPOCH

result_path = result_path + 'model_' + NAME + '/'

#################################### Data Loading #######################################

DATA_DIR = '../../../Data/Datasets/Final_Data/'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_language.py!')

seqs, folds, folds_dict, charmap, inv_charmap = data_helpers.load_dataset_protein( #MK change
    max_length=SEQ_LEN,
    max_n_examples=MAX_N_EXAMPLES,
    data_dir=DATA_DIR
)

f_s_dic = DataLoading.Train_dic(DATA_DIR + 'fold_train',DATA_DIR + 'seq_train') #SZ add
inter_dic = DataLoading.Interval_dic(DATA_DIR + 'Interval_1.fa') #SZ add
fold_train = DataLoading.file_list(DATA_DIR + 'unique_fold_train') #SZ add
fold_vali = DataLoading.file_list(DATA_DIR + 'fold_val') #SZ add
fold_test = DataLoading.file_list(DATA_DIR + 'fold_test')  #SZ add

Class_info = {'a':{},'b':{},'c':{},'d':{},'e':{},'f':{},'g':{}}

for f in inter_dic.keys():
    l_min = 10000
    l_max = 0
    for s in inter_dic[f][1:]:
        l = len(s)
        if l > l_max:
            l_max = l
        if l < l_min:
            l_min = l
    Class_info[f[0]][f] = [l_min,l_max]

for f in inter_dic.keys():
    if inter_dic[f][0] > 50:
        Class_info[f[0]][f].append('easy')
    elif inter_dic[f][0] > 5:
        Class_info[f[0]][f].append('medium')
    else:
        Class_info[f[0]][f].append('hard')

print 'Data loading successfully!'

classes = ['a','b','c','d','e','f','g','easy','medium','hard']

######################################### Training Set #########################################

if train_flag:

    os.system('mkdir ' + result_path + 'train')
    YieldRatio_Dict_train = {'a':[],'b':[],'c':[],'d':[],'e':[],'f':[],'g':[],'easy':[],'medium':[],'hard':[]}
    best_fold_train = {'a':[0,''],'b':[0,''],'c':[0,''],'d':[0,''],'e':[0,''],'f':[0,''],'g':[0,''],'easy':[0,''],'medium':[0,''],'hard':[0,'']}
    YieldRatio_train = []
    ratio_train = {}
    yrf_train = open(result_path + 'train/yield_ratio_fold_' + NAME + '_fold_train.fa','w')
    yrf_train.write('fold\trank\tYR\tmin_len\tmax_len\tsuc_num\tgen_num\ttime')
    stat_train = open(result_path + 'train/yield_ratio_stat_' + NAME + '_fold_train.fa','w')
    yr_sort_train = open(result_path + 'train/yield_ratio_sort_' + NAME + '_fold_train.fa','w')

    for fo in fold_train:
        fold_data = DataLoading.columns_to_lists(result_path + fo)
        ratio = float(fold_data[0][0])
        suc_num = int(fold_data[1][0])
        gen_num = int(fold_data[2][0])
        time = float(fold_data[3][0])
        
        min_len = Class_info[fo[0]][fo][0]
        max_len = Class_info[fo[0]][fo][1]
        rank = Class_info[fo[0]][fo][2]

        if ratio in ratio_train.keys():
            ratio_train[ratio].append(fo)
        else:
            ratio_train[ratio] = [fo]

        YieldRatio_train.append(ratio)
        YieldRatio_Dict_train[fo[0]].append(ratio)
        YieldRatio_Dict_train[rank].append(ratio)

        yrf_train.write(fo + '\t' + rank + '\t' + str(ratio) + '\t')
        yrf_train.write(str(min_len) + '\t' + str(max_len) + '\t')
        yrf_train.write(str(suc_num) + '\t' + str(gen_num) + '\t' + str(time) + '\n')
    
        if ratio > best_fold_train[fo[0]][0]:
            best_fold_train[fo[0]] = [ratio,fo]
        elif ratio == best_fold_train[fo[0]][0]:
            best_fold_train[fo[0]].append(fo)

        if ratio > best_fold_train[rank][0]:
            best_fold_train[rank] = [ratio,fo]
        elif ratio == best_fold_train[rank][0]:
            best_fold_train[rank].append(fo)

    average_yr_train = np.mean(YieldRatio_train)

    stat_train.write('Average Yield Ratio: ' + str(average_yr_train) + '\n')
    stat_train.write('\n')
    stat_train.write('class' + '\t' + 'aver_yr' + '\t' + 'size'+ '\t' + 'best_fold' + '\t' + 'best_yr' + '\n')

    for cl in classes:
        class_aver_ratio = np.mean(YieldRatio_Dict_train[cl])
        stat_train.write(cl + '\t' + str(class_aver_ratio) + '\t' + str(len(YieldRatio_Dict_train[cl])) + '\t')
        for fol in best_fold_train[cl][1:]:
            stat_train.write(fol + '\t')
            stat_train.write(str(best_fold_train[cl][0]) + '\t')
        stat_train.write('\n')

    ratio_sorted = sorted(ratio_train.keys(),reverse = True)
    for yr in ratio_sorted:
        for fol in ratio_train[yr]:
            yr_sort_train.write(fol + '\t' + Class_info[fol[0]][fol][2] + '\t' + str(yr) + '\n')

    yrf_train.close()
    stat_train.close()
    yr_sort_train.close()

######################################### Test Set ############################################

if test_flag:

    os.system('mkdir ' + result_path + 'test')
    YieldRatio_Dict_test = {'a':[],'b':[],'c':[],'d':[],'e':[],'f':[],'g':[],'easy':[],'medium':[],'hard':[]}
    best_fold_test = {'a':[0,''],'b':[0,''],'c':[0,''],'d':[0,''],'e':[0,''],'f':[0,''],'g':[0,''],'easy':[0,''],'medium':[0,''],'hard':[0,'']}
    YieldRatio_test = []
    ratio_test = {}
    yrf_test = open(result_path + 'test/yield_ratio_fold_' + NAME + '_fold_test.fa','w')
    yrf_test.write('fold\trank\tYR\tmin_len\tmax_len\tsuc_num\tgen_num\ttime')
    stat_test = open(result_path + 'test/yield_ratio_stat_' + NAME + '_fold_test.fa','w')
    yr_sort_test = open(result_path + 'test/yield_ratio_sort_' + NAME + '_fold_test.fa','w')

    for fo in fold_test:
        fold_data = DataLoading.columns_to_lists(result_path + fo)
        ratio = float(fold_data[0][0])
    	suc_num = int(fold_data[1][0])
        gen_num = int(fold_data[2][0])
        time = float(fold_data[3][0])

        min_len = Class_info[fo[0]][fo][0]
        max_len = Class_info[fo[0]][fo][1]
        rank = Class_info[fo[0]][fo][2]

        if ratio in ratio_test.keys():
            ratio_test[ratio].append(fo)
        else:
            ratio_test[ratio] = [fo]

        YieldRatio_test.append(ratio)
        YieldRatio_Dict_test[fo[0]].append(ratio)
        YieldRatio_Dict_test[rank].append(ratio)

        yrf_test.write(fo + '\t' + rank + '\t' + str(ratio) + '\t')
        yrf_test.write(str(min_len) + '\t' + str(max_len) + '\t')
        yrf_test.write(str(suc_num) + '\t' + str(gen_num) + '\t' + str(time) + '\n')

        if ratio > best_fold_test[fo[0]][0]:
            best_fold_test[fo[0]] = [ratio,fo]
        elif ratio == best_fold_test[fo[0]][0]:
            best_fold_test[fo[0]].append(fo)

        if ratio > best_fold_test[rank][0]:
            best_fold_test[rank] = [ratio,fo]
        elif ratio == best_fold_test[rank][0]:
            best_fold_test[rank].append(fo)

    average_yr_test = np.mean(YieldRatio_test)

    stat_test.write('Average Yield Ratio: ' + str(average_yr_test) + '\n')
    stat_test.write('\n')
    stat_test.write('class' + '\t' + 'aver_yr' + '\t' + 'size'+ '\t' + 'best_fold' + '\t' + 'best_yr' + '\n')

    for cl in classes:
        class_aver_ratio = np.mean(YieldRatio_Dict_test[cl])
        stat_test.write(cl + '\t' + str(class_aver_ratio) + '\t' + str(len(YieldRatio_Dict_test[cl])) + '\t')
        for fol in best_fold_test[cl][1:]:
            stat_test.write(fol + '\t')
            stat_test.write(str(best_fold_test[cl][0]) + '\t')
        stat_test.write('\n')

    ratio_sorted = sorted(ratio_test.keys(),reverse = True)
    for yr in ratio_sorted:
        for fol in ratio_test[yr]:
            yr_sort_test.write(fol + '\t' + Class_info[fol[0]][fol][2] + '\t' + str(yr) + '\n')

    yrf_test.close()
    stat_test.close()
    yr_sort_test.close()

######################################### Validation Set #########################################

if vali_flag:

    os.system('mkdir ' + result_path + 'vali')
    YieldRatio_Dict_vali = {'a':[],'b':[],'c':[],'d':[],'e':[],'f':[],'g':[],'easy':[],'medium':[],'hard':[]}
    best_fold_vali = {'a':[0,''],'b':[0,''],'c':[0,''],'d':[0,''],'e':[0,''],'f':[0,''],'g':[0,''],'easy':[0,''],'medium':[0,''],'hard':[0,'']}
    YieldRatio_vali = []
    ratio_vali = {}
    yrf_vali = open(result_path + 'vali/yield_ratio_fold_' + NAME + '_fold_vali.fa','w')
    yrf_vali.write('fold\trank\tYR\tmin_len\tmax_len\tsuc_num\tgen_num\ttime')
    stat_vali = open(result_path + 'vali/yield_ratio_stat_' + NAME + '_fold_vali.fa','w')
    yr_sort_vali = open(result_path + 'vali/yield_ratio_sort_' + NAME + '_fold_vali.fa','w')

    for fo in fold_vali:
        fold_data = DataLoading.columns_to_lists(result_path + fo)
        ratio = float(fold_data[0][0])
        suc_num = int(fold_data[1][0])
        gen_num = int(fold_data[2][0])
        time = float(fold_data[3][0])

        min_len = Class_info[fo[0]][fo][0]
        max_len = Class_info[fo[0]][fo][1]
        rank = Class_info[fo[0]][fo][2]

        if ratio in ratio_vali.keys():
            ratio_vali[ratio].append(fo)
        else:
            ratio_vali[ratio] = [fo]

        YieldRatio_vali.append(ratio)
        YieldRatio_Dict_vali[fo[0]].append(ratio)
        YieldRatio_Dict_vali[rank].append(ratio)

        yrf_vali.write(fo + '\t' + rank + '\t' + str(ratio) + '\t')
        yrf_vali.write(str(min_len) + '\t' + str(max_len) + '\t')
        yrf_vali.write(str(suc_num) + '\t' + str(gen_num) + '\t' + str(time) + '\n')

        if ratio > best_fold_vali[fo[0]][0]:
            best_fold_vali[fo[0]] = [ratio,fo]
        elif ratio == best_fold_vali[fo[0]][0]:
            best_fold_vali[fo[0]].append(fo)

        if ratio > best_fold_vali[rank][0]:
            best_fold_vali[rank] = [ratio,fo]
        elif ratio == best_fold_vali[rank][0]:
            best_fold_vali[rank].append(fo)

    average_yr_vali = np.mean(YieldRatio_vali)

    stat_vali.write('Average Yield Ratio: ' + str(average_yr_vali) + '\n')
    stat_vali.write('\n')
    stat_vali.write('class' + '\t' + 'aver_yr' + '\t' + 'size'+ '\t' + 'best_fold' + '\t' + 'best_yr' + '\n')

    for cl in classes:
        class_aver_ratio = np.mean(YieldRatio_Dict_vali[cl])
        stat_vali.write(cl + '\t' + str(class_aver_ratio) + '\t' + str(len(YieldRatio_Dict_vali[cl])) + '\t')
        for fol in best_fold_vali[cl][1:]:
            stat_vali.write(fol + '\t')
            stat_vali.write(str(best_fold_vali[cl][0]) + '\t')
        stat_vali.write('\n')

    ratio_sorted = sorted(ratio_vali.keys(),reverse = True)
    for yr in ratio_sorted:
        for fol in ratio_vali[yr]:
            yr_sort_vali.write(fol + '\t' + Class_info[fol[0]][fol][2] + '\t' + str(yr) + '\n')

    yrf_vali.close()
    stat_vali.close()
    yr_sort_vali.close()
