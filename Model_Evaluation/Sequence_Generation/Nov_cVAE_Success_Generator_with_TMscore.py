################################ Import packages #################################

import os, sys
sys.path.append(os.getcwd())

import time
import numpy as np
import data_helpers #SZ change
import DataLoading #SZ add
import os

from keras.models import model_from_json
from keras.engine.topology import Layer
import theano.tensor as T
from keras import backend as K
from keras.constraints import maxnorm

from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Merge, Convolution1D
from keras.layers.normalization import BatchNormalization

############################## Set paths and parameters ################################

GEN_NUM = int(sys.argv[1])
#MIN_LEN = int(sys.argv[4])
#MAX_LEN = int(sys.argv[5])

if not os.path.exists('Pipeline_Sample'):
    os.system('mkdir Pipeline_Sample')

FOLD = 'nov'
fold_name = 'nov'

DATA_DIR = '../../Data/Datasets/Final_Data/'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_language.py!')

THRESHOLD = 0.21

BATCH_SIZE = 10000 # Batch size
SEQ_LEN = 160 # Sequence length in characters
DIM = 512 # Model dimensionality. This is fairly slow and overfits, even on
          # Billion Word. Consider decreasing for smaller datasets.
LAMBDA = 10 # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 50000 # Max number of data examples to load. If data loading
                          # is too slow or takes too much RAM, you can decrease
                          # this (at the expense of having less training data).
TOP_NUM = 10

################################ Load Data #################################

seqs, folds, folds_dict, charmap, inv_charmap = data_helpers.load_dataset_protein( #MK change
    max_length=SEQ_LEN,
    max_n_examples=MAX_N_EXAMPLES,
    data_dir=DATA_DIR
)

novel_array = np.loadtxt(DATA_DIR + 'novel_coordinate')

print 'Novel fold coordinates:',novel_array
#print 'Data loading successfully!'

################################ DeepSF ###########################

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

def create_aa_feature(inp,num):

    num_class = 20
    output = np.zeros((num,SEQ_LEN,num_class))
    for i in range(num):
        output[i,:,:] = np.eye(num_class)[inp[i,]-1]

    return output

model_file="./DeepSF_model_weight_more_folds/model-train-DLS2F.json"
model_weight="./DeepSF_model_weight_more_folds/model-train-weight-DLS2F.h5"
deepsf_fold="./DeepSF_model_weight_more_folds/fold_label_relation2.txt"
kmaxnode=30

json_file_model = open(model_file, 'r')
loaded_model_json = json_file_model.read()
json_file_model.close()
DLS2F_CNN = model_from_json(loaded_model_json, custom_objects={'K_max_pooling1d': K_max_pooling1d})
DLS2F_CNN.load_weights(model_weight)
DLS2F_CNN.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer="nadam")

fold_index = DataLoading.Accuracy_index(path = 'DeepSF_model_weight_more_folds/fold_label_relation2.txt')

print 'Data loading successfully!'

################################ Generate sequences #################################

def generate_samples(fold,size,f_name):
    temp_index = 0
    temp_file_name = 'Temp_' + f_name + '_' + str(temp_index)
    while os.path.exists(temp_file_name):
        temp_index += 1
        temp_file_name = 'Temp_' + f_name + '_' + str(temp_index)
    command = './cVAE_seq_helper.sh %s %s %s'%(fold,str(size),temp_file_name)
    os.system(command)
    with open(temp_file_name,'r') as temp_file:
        lines = temp_file.readlines()
    gen_seq = []
    gen_num = 0
    for i in lines:
        gen_num += 1
        seq = i.strip('\n').lower()
        if not 'x' in seq:
            gen_seq.append(seq)
    
    os.system('rm ' + temp_file_name)
    
    if gen_num != size:
        print 'Generation Error! The amount is not correct! %d and %d'%(gen_num,size)
        return None,None
    
    return gen_seq,gen_num

all_file = open('Pipeline_Sample/cVAE_all_backup_' + str(GEN_NUM) + '_' + fold_name + '_with_TMscore','w')
success_file = open('Pipeline_Sample/cVAE_Fasta_' + str(GEN_NUM) + '_success_' + fold_name + '_with_TMscore','w')
all_file.close()
success_file.close()

s_index = 0
batch_num = 0
valid_batch_num = 0
gen_all_num = 0
gen_noX_num = 0

while(s_index < GEN_NUM):
    
    batch_num += 1    
    #print batch_num    

    S_test,gen_batch_num = generate_samples(FOLD,BATCH_SIZE,fold_name)
    if S_test != None:
        valid_batch_num += 1
        V_SIZE = len(S_test)   
        gen_all_num += gen_batch_num
        gen_noX_num += V_SIZE
    else:
        V_SIZE = -1
   
    if V_SIZE > 0: 

        all_file = open('Pipeline_Sample/cVAE_all_backup_' + str(GEN_NUM) + '_' + fold_name + '_with_TMscore','a')
        success_file = open('Pipeline_Sample/cVAE_Fasta_' + str(GEN_NUM) + '_success_' + fold_name + '_with_TMscore','a')

        test_se = [s+'!'*(SEQ_LEN - len(s)) for s in S_test]
        test_se = [tuple(s) for s in test_se]
        test_seq = create_aa_feature(np.asarray([[charmap[c] for c in l] for l in test_se]).reshape((V_SIZE,SEQ_LEN)),V_SIZE)
        prediction= DLS2F_CNN.predict([test_seq])
        top_prediction=prediction.argsort()[:,::-1][:,:TOP_NUM]
        
        for p in range(V_SIZE):
            
            all_file.write(S_test[p] + '\n')            

            f_pre = [fold_index[i] for i in top_prediction[p]]
             
            if ('a.30' in f_pre) or ('a.60' in f_pre) or ('a.53' in f_pre):
                s_index += 1
                if s_index <= GEN_NUM:
                    success_file.write('>' + str(s_index) + '\n')
                    success_file.write(S_test[p].upper() + '\n')

            #for f_p in f_pre:
            #    if f_p in folds_dict.keys():
            #        if np.linalg.norm(novel_array - np.array(folds_dict[f_p])) <= THRESHOLD:
            #            s_index += 1
            #            if s_index <= GEN_NUM:
            #                success_file.write('>' + str(s_index) + '\n')
            #                success_file.write(S_test[p].upper() + '\n')
    
        all_file.write('\n')
        all_file.close()
        success_file.close()

    print batch_num, valid_batch_num, gen_batch_num, V_SIZE
        

print '%d batch of cVAE sequences generated and %d are valid.'%(batch_num,valid_batch_num)
print '%d cVAE sequences generated.'%gen_all_num
print '%d cVAE sequences without X generated.'%gen_noX_num
print '%d successful cVAE sequences generated.'%s_index
