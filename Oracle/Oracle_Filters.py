import os
import numpy as np
from keras.models import model_from_json
from keras.engine.topology import Layer
import theano.tensor as T
from keras import backend as K
from keras.constraints import maxnorm

from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Merge, Convolution1D
from keras.layers.normalization import BatchNormalization

################## For Modified DeepSF ##################################

def create_aa_feature(inp,num,SEQ_LEN):
    num_class = 20
    output = np.zeros((num,SEQ_LEN,num_class))
    for i in range(num):
        output[i,:,:] = np.eye(num_class)[inp[i,]-1]
    return output

################## For Original DeepSF ##################################

def import_DLS2FSVM(filename, delimiter='\t', delimiter2=' ',comment='>',skiprows=0, start=0, end = 0,target_col = 1, dtype=np.float32):
    # Open a file
    file = open(filename, "r")
    #print "Name of the file: ", file.name
    if skiprows !=0:
       dataset = file.read().splitlines()[skiprows:]
    if skiprows ==0 and start ==0 and end !=0:
       dataset = file.read().splitlines()[0:end]
    if skiprows ==0 and start !=0:
       dataset = file.read().splitlines()[start:]
    if skiprows ==0 and start !=0 and end !=0:
       dataset = file.read().splitlines()[start:end]
    else:
       dataset = file.read().splitlines()
    #print dataset
    newdata = []
    for i in range(0,len(dataset)):
        line = dataset[i]
        if line[0] != comment:
           temp = line.split(delimiter,target_col)
           feature = temp[target_col]
           label = temp[0]
           if label == 'N':
               label = 0
           fea = feature.split(delimiter2)
           newline = []
           newline.append(int(label))
           for j in range(0,len(fea)):
               if fea[j].find(':') >0 :
                   (num,val) = fea[j].split(':')
                   newline.append(float(val))

           newdata.append(newline)
    data = np.array(newdata, dtype=dtype)
    file.close()
    return data

def Seq_Fasta(seq,path):
    """
    Record the sequence in a file with Fasta format.
    """
    name = path.split('/')[-1].split('.')[0]
    with open(path,'w') as fasta_file:
        fasta_file.write('>' + name + '\n')
        fasta_file.write(seq.upper() + '\n')
    return 0

def PSSM(seq_file,database,path,num_threads=4,comp_based_stats=1,evalue=0.001,num_iter=1):
    """
    Calculate the PSSM according to the given fasta file with psiblast.
    """
    if not path.endswith('/'):
        path += '/'
    seq_name = seq_file.split('/')[-1].split('.')[0]
    #print seq_name
    command = 'psiblast -query ' + seq_file
    command += ' -out ' + path + seq_name + '.out'
    command += ' -num_threads ' + str(num_threads)
    command += ' -comp_based_stats ' + str(comp_based_stats)
    command += ' -evalue ' + str(evalue)
    command += ' -db ' + database
    command += ' -num_iterations ' + str(num_iter)
    command += ' -out_ascii_pssm ' + path + seq_name + '_pssm' 
    command += ' -save_pssm_after_last_round'
    os.system(command)
    
    while not os.path.exists(path + seq_name + '_pssm'):
        continue

    pssm_file = open(path + seq_name + '_pssm','r')
    pssm_lines = pssm_file.readlines()
    pssm_file.close()

    pssm_value_1 = []

    for line in pssm_lines[3:]:
        if line == '\n':
            break
        else:
            line = [ch for ch in line.strip('\n').split(' ') if ch != '']
            line_1 = [int(i) for i in line[2:22]]
            pssm_value_1 += line_1

    pssm_index_1 = 0
    pssm_deepsf_line_1 = ''
    for v in pssm_value_1:
        pssm_index_1 += 1
        pssm_deepsf_line_1 += str(pssm_index_1) + ':' + str(v) + ' '
    pssm_deepsf_line_1 = pssm_deepsf_line_1.strip(' ') + '\n'

    pssm_deepsf_1 = open(path + seq_name + '.pssm_fea','w')
    pssm_deepsf_1.write('>' + seq_name + '\n0\t')
    pssm_deepsf_1.write(pssm_deepsf_line_1)
    pssm_deepsf_1.close()

    return 0


def PSSM_pgp(seq_file,database,path,blastpgp_path,j=2,e=0.001,h=1e-10):
    """
    Calculate the PSSM according to the given fasta file with blastpgp.
    """
    if not path.endswith('/'):
        path += '/'
    seq_name = seq_file.split('/')[-1].split('.')[0]
    #print seq_name
    command = blastpgp_path + ' -i ' + seq_file
    command += ' -o ' + path + seq_name + '.blastpgp'
    command += ' -j ' + str(j)
    command += ' -e ' + str(e)
    command += ' -h ' + str(h)
    command += ' -d ' + database
    command += ' -Q ' + path + seq_name + '.pssm'
    os.system(command)

    while not os.path.exists(path + seq_name + '.pssm'):
        continue

    pssm_file = open(path + seq_name + '.pssm','r')
    pssm_lines = pssm_file.readlines()
    pssm_file.close()

    pssm_value_1 = []

    for line in pssm_lines[3:]:
        if line == '\n':
            break
        else:
            line = [ch for ch in line.strip('\n').split(' ') if ch != '']
            line_1 = [int(i) for i in line[2:22]]
            pssm_value_1 += line_1

    pssm_index_1 = 0
    pssm_deepsf_line_1 = ''
    for v in pssm_value_1:
        pssm_index_1 += 1
        pssm_deepsf_line_1 += str(pssm_index_1) + ':' + str(v) + ' '
    pssm_deepsf_line_1 = pssm_deepsf_line_1.strip(' ') + '\n'

    pssm_deepsf_1 = open(path + seq_name + '.pssm_fea','w')
    pssm_deepsf_1.write('>' + seq_name + '\n0\t')
    pssm_deepsf_1.write(pssm_deepsf_line_1)
    pssm_deepsf_1.close()

    return 0


def one_hot(seq,dic):
    wide = len(dic.keys())
    result = []
    for ch in seq:
        row = [0]*wide
        row[dic[ch]] = 1
        result.append(row)
    return result

def AA_SS_SA(seq_file,scratch_path,path,AA_dic,SS_dic,SA_dic):
    """
    Calculate the AA_SS_SA according to the given fasta file.
    """
    if not path.endswith('/'):
        path += '/'
    file_name = seq_file.split('/')[-1].split('.')[0]
    command = scratch_path + ' ' + seq_file
    command += ' ' + path + file_name
    
    os.system(command)

    seq_dic = {}

    ################################# AA ################################ 
    with open(seq_file,'r') as aa_file:
        aa_lines = aa_file.readlines()

    SEQ_num = 0
    for line in aa_lines:
        if '>' in line:
            if SEQ_num != 0:
                seq_len = len(seq)
                seq_dic[SEQ_name] = [seq,seq_len]
            SEQ_num += 1
            SEQ_name = line.strip('\n').split(' ')[0][1:]
            seq = ''
        else:
            seq += line.strip('\n')

    seq_len = len(seq)
    seq_dic[SEQ_name] = [seq,seq_len]

    while not (os.path.exists(path + file_name + '.ss') and os.path.exists(path + file_name + '.acc')):
        continue

    ################################# SS ################################
    with open(path + file_name + '.ss','r') as ss_file:
        ss_lines = ss_file.readlines()

    SS_num = 0
    for line in ss_lines:
        if '>' in line:
            if SS_num != 0:
                seq_dic[SEQ_name].append(ss)
                if seq_dic[SEQ_name][1] != len(ss):
                    print 'Error! Length of AA and SS so not match!'
            SS_num += 1
            SEQ_name = line.strip('\n').split(' ')[0][1:]
            ss = ''
        else:
            ss += line.strip('\n')

    seq_dic[SEQ_name].append(ss)
    if seq_dic[SEQ_name][1] != len(ss):
        print 'Error! Length of AA and SS so not match!'
    ################################# SA ################################
    with open(path + file_name + '.acc','r') as sa_file:
        sa_lines = sa_file.readlines()

    SA_num = 0
    for line in sa_lines:
        if '>' in line:
            if SA_num != 0:
                seq_dic[SEQ_name].append(sa)
                if seq_dic[SEQ_name][1] != len(sa):
                    print 'Error! Length of AA and SA so not match!'
            SA_num += 1
            SEQ_name = line.strip('\n').split(' ')[0][1:]
            sa = ''
        else:
            sa += line.strip('\n')

    seq_dic[SEQ_name].append(sa)
    if seq_dic[SEQ_name][1] != len(sa):
        print 'Error! Length of AA and SA so not match!'
    ######################### Write the feature file #####################
    for seq_name in seq_dic.keys():
        feature = '0\t'
        index = 0
        seq_len = seq_dic[seq_name][1]
        aa_list = one_hot(seq_dic[seq_name][0],AA_dic)
        ss_list = one_hot(seq_dic[seq_name][2],SS_dic)
        sa_list = one_hot(seq_dic[seq_name][3],SA_dic)
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

        feature_deepsf = open(path + seq_name + '.fea_aa_ss_sa','w')
        feature_deepsf.write('>' + seq_name + '\n')
        feature_deepsf.write(feature.strip(' ') + '\n')
        feature_deepsf.close()

    return 0

########################### For Both ##################################

class K_max_pooling1d(Layer):
    def __init__(self,  ktop, **kwargs):
        self.ktop = ktop
        super(K_max_pooling1d, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],self.ktop,input_shape[2])

    def call(self,x,mask=None):
        output = x[T.arange(x.shape[0]).dimshuffle(0, "x", "x"),
              T.sort(T.argsort(x, axis=1)[:, -self.ktop:, :], axis=1),
              T.arange(x.shape[2]).dimshuffle("x", "x", 0)]
        return output

    def get_config(self):
        config = {'ktop': self.ktop}
        base_config = super(K_max_pooling1d, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



