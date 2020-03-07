import os, sys

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

model_file= "DeepSF_model_weight_more_folds/model-train-DLS2F.json"
model_weight="DeepSF_model_weight_more_folds/model-train-weight-DLS2F.h5"
deepsf_fold="DeepSF_model_weight_more_folds/fold_label_relation2.txt"
kmaxnode=30

json_file_model = open(model_file, 'r')
loaded_model_json = json_file_model.read()
json_file_model.close()
DLS2F_CNN = model_from_json(loaded_model_json, custom_objects={'K_max_pooling1d': K_max_pooling1d})
DLS2F_CNN.load_weights(model_weight)
DLS2F_CNN.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer="nadam")

fold_index = DataLoading.Accuracy_index(path = 'DeepSF_model_weight_more_folds/fold_label_relation2.txt')

def file_list(path):
    fil = open(path,'r')
    lines = fil.readlines()
    lis = []
    for line in lines:
        if line != '\n':# and not ('X' in line):
            lis.append(line.strip('\n').lower())
    fil.close()
    return lis

gen_num = 0
suc_num = 0

with tf.Session() as session:

    session.run(tf.initialize_all_variables())
    
    #S_list = file_list(SAMPLE_PATH + FOLD)
    while(len(S_list) > BATCH_SIZE):
        S_select = S_list[0:BATCH_SIZE]
        S_list = S_list[BATCH_SIZE:]
        S_test = []
        for seq in S_select:
            if not ('x' in seq):
                S_test.append(seq)
        TEST_SIZE = len(S_test)
        gen_num += TEST_SIZE
        
        test_se = [s+'!'*(SEQ_LEN - len(s)) for s in S_test]
        test_se = [tuple(s) for s in test_se]
        test_seq = create_aa_feature(np.asarray([[charmap[c] for c in l] for l in test_se]).reshape((TEST_SIZE,SEQ_LEN)),TEST_SIZE)
        prediction= DLS2F_CNN.predict([test_seq])
        top10_prediction=prediction.argsort()[:,::-1][:,:10]
        for p in top10_prediction:
            f_pre = [fold_index[i] for i in p]
            if FOLD in f_pre:
                suc_num += 1
        if gen_num >= BATCH_SIZE and (suc_num >= SUC_SIZE or gen_num >= GEN_SIZE):
                break

    ratio = float(suc_num)/float(gen_num)

    
    #print FOLD    
    #print suc_num
    #print gen_num
    print ratio

