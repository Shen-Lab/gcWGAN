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

fold_len = 20
BATCH_SIZE = 1400 # Batch size
SUCC_NUM = 10
GEN_NUM = 1000
GEN_UPPER = 100000
SEQ_LEN = 160 # Sequence length in characters
DIM = 512 # Model dimensionality. This is fairly slow and overfits, even on
          # Billion Word. Consider decreasing for smaller datasets.
LAMBDA = 10 # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 50000 # Max number of data examples to load. If data loading
                          # is too slow or takes too much RAM, you can decrease
                          # this (at the expense of having less training data).

PATH = sys.argv[1]
EPOCH = sys.argv[2]
FOLD = sys.argv[3]

if 'gcWGAN' in PATH:
    result_path = '../../../Results/Accuracy/Yield_Ratio_Result/gcWGAN/' 
    sample_path = 'Yield_Ratio_Samples/gcWGAN_Samples/'
elif 'cWGAN' in PATH:
    result_path = '../../../Results/Accuracy/Yield_Ratio_Result/cWGAN/'
    sample_path = 'Yield_Ratio_Samples/cWGAN_Samples/'
else:
    print 'Path Error!'
    quit()

if not os.path.exists('../../../Results'):
    os.system('mkdir ../../../Results')
if not os.path.exists('../../../Results/Accuracy'):
    os.system('mkdir ../../../Results/Accuracy')
if not os.path.exists('../../../Results/Accuracy/Yield_Ratio_Result'):
    os.system('mkdir ../../../Results/Accuracy/Yield_Ratio_Result')
if not os.path.exists(result_path):
    os.system('mkdir ' + result_path)

if not os.path.exists('Yield_Ratio_Samples'):
    os.system('mkdir Yield_Ratio_Samples')
if not os.path.exists(sample_path):
    os.system('mkdir ' + sample_path)

if PATH[-1] == '/':
    PATH = PATH.strip('/')

path_split = PATH.split('_')
NAME = ''
flag = 0
"""
for j in path_split:
    if flag == 1:
        NAME += j
        flag = 2
    elif flag == 2:
        NAME += '_' + j
    if (('oint' in j) or ('OINT' in j)) and flag == 0:
        flag = 1
        p_index = path_split.index(j)
        print j
        print p_index
"""
for j in path_split:
    if not ('heck' in j or 'o' in j or 'HECK' in j or 'O' in j):
        NAME += j + '_'
        if '.' in j and flag == 0:
            flag = 1
            p_index = path_split.index(j)
            print j
            print p_index

NAME += EPOCH

#print NAME

#if not os.path.exists('mkdir Yield_Ratio_Result'):
#    os.system('mkdir Yield_Ratio_Result')
#if not os.path.exists('mkdir Yield_Ratio_Result/model_' + NAME):
#    os.system('mkdir Yield_Ratio_Result/model_' + NAME)
#if not os.path.exists('mkdir Yield_Ratio_Result/model_' + NAME + '/Samples'):
#    os.system('mkdir Yield_Ratio_Result/model_' + NAME + '/Samples')

result_path = result_path + 'model_' + NAME + '/' 
sample_path = sample_path + 'model_' + NAME + '/'

if not os.path.exists(result_path):
    os.system('mkdir ' + result_path)
if not os.path.exists(sample_path): 
    os.system('mkdir ' + sample_path)

if flag == 0:
    noise_len = 128
    CRITIC_ITERS = 20
else:
    noise_len = int(path_split[p_index + 2])
    CRITIC_ITERS = int(path_split[p_index + 1])

#################################### Data Loading #######################################

DATA_DIR = '../../../Data/Datasets/Final_Data/'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_language.py!')

seqs, folds, folds_dict, charmap, inv_charmap = data_helpers.load_dataset_protein( #MK change
    max_length=SEQ_LEN,
    max_n_examples=MAX_N_EXAMPLES,
    data_dir=DATA_DIR
)

fold_index = DataLoading.Accuracy_index(path = 'DeepSF_model_weight_more_folds/fold_label_relation2.txt')

print 'Data loading successfully!'

check_dic = {}
f_list = os.listdir(PATH)
for f in f_list:
    f = f.split('.')[0].split('_')
    if f[0] == 'model':
        check_dic[f[1]] = f[2]

check_index = sorted([int(i) for i in check_dic.keys()])

print 'Load the index of check points successfully!'

#################################### DeepSF ############################################

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

############################################ Model ####################################

def softmax(logits):
    return tf.reshape(
        tf.nn.softmax(
            tf.reshape(logits, [-1, len(charmap)])
        ),
        tf.shape(logits)
    )

def make_noise(shape):
    return tf.random_normal(shape)

def ResBlock(name, inputs):
    output = inputs
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.1', DIM, DIM, 5, output)
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.2', DIM, DIM, 5, output)
    return inputs + (0.3*output)

def ResBlock_v2(name, inputs,size): #MK add
    output = inputs
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.1', size, size, 5, output)
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.2', size, size, 5, output)
    return inputs + (0.3*output)

def Generator(n_samples, labels, prev_outputs=None): #MK change
    output = make_noise(shape=[n_samples, noise_len])
    output = tf.concat([output,labels],axis=1) #MK add
    output = lib.ops.linear.Linear('Generator.Input', noise_len+fold_len, SEQ_LEN*DIM, output) #MK change
    output = tf.reshape(output, [-1, DIM, SEQ_LEN])
    output = ResBlock('Generator.1', output)
    output = ResBlock('Generator.2', output)
    output = ResBlock('Generator.3', output)
    output = ResBlock('Generator.4', output)
    output = ResBlock('Generator.5', output)
    output = lib.ops.conv1d.Conv1D('Generator.Output', DIM, len(charmap), 1, output)
    output = tf.transpose(output, [0, 2, 1])
    output = softmax(output)
    return output

def Discriminator(inputs,labels): #MK change
    output = tf.transpose(inputs, [0,2,1])
    output = lib.ops.conv1d.Conv1D('Discriminator.Input', len(charmap), DIM, 1, output)
    output = ResBlock('Discriminator.1', output)
    output = ResBlock('Discriminator.2', output)
    output = ResBlock('Discriminator.3', output)
    output = ResBlock('Discriminator.4', output)
    output = ResBlock('Discriminator.5', output)
    output = tf.reshape(output, [-1, SEQ_LEN*DIM])
    size= 100 #MK add
    output = lib.ops.linear.Linear('Discriminator.reduction', SEQ_LEN*DIM,size, output) #MK change
    output = tf.concat([output,labels],axis=1) #MK add
    output = tf.contrib.layers.fully_connected(output,300,scope='Discriminator.fully',reuse=tf.AUTO_REUSE) #MK add
    output = lib.ops.linear.Linear('Discriminator.output',300 , 1, output) #MK add
    return output

real_inputs_discrete = tf.placeholder(tf.int32, shape=[BATCH_SIZE, SEQ_LEN])
real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
real_inputs_label = tf.placeholder(tf.float32, shape=[BATCH_SIZE, fold_len]) #MK add
fake_inputs = Generator(BATCH_SIZE,real_inputs_label) #MK change
fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)

disc_real = Discriminator(real_inputs,real_inputs_label) #MK change 
disc_fake = Discriminator(fake_inputs,real_inputs_label) #MK change

disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = -tf.reduce_mean(disc_fake)

# WGAN lipschitz-penalty
alpha = tf.random_uniform(
    shape=[BATCH_SIZE,1,1], 
    minval=0.,
    maxval=1.
)
differences = fake_inputs - real_inputs
interpolates = real_inputs + (alpha*differences)
gradients = tf.gradients(Discriminator(interpolates,real_inputs_label), [interpolates])[0] #MK change
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)

######################################## Calculation ####################################

saver  = tf.train.Saver()

with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    def generate_samples(label): #MK change
        samples = session.run(fake_inputs,feed_dict={real_inputs_label:label}) #MK change
        samples = np.argmax(samples, axis=2)
        decoded_samples = []
        for i in xrange(len(samples)):
            decoded = []
            for j in xrange(len(samples[i])):
                decoded.append(inv_charmap[samples[i][j]])
            decoded_samples.append(tuple(decoded))
        return decoded_samples
    
    check_select = [EPOCH]
    select_index = 2

    for c in check_select:
        check_p = c
        print c
        print check_dic[str(c)]
        saver.restore(session,PATH + "/model_"+str(c)+"_" + check_dic[str(c)] + ".ckpt")
        print 'Restore Successfully!'
        
        yr_fold_file = open(result_path + FOLD,'w')
        yr_sample_file = open(sample_path + 'sample_' + NAME + '_' + FOLD,'w')
        yr_succ_file = open(sample_path + 'success_sample_' + NAME + '_' + FOLD,'w')
        yr_sample_file.close()
        yr_succ_file.close()
      
        start = time.time()
        suc_num = 0
        gen_num = 0
        samples_f = []
        test_se = []

        while((suc_num < SUCC_NUM) or (gen_num < GEN_NUM)):
            f_batch = [folds_dict[FOLD]] * BATCH_SIZE
            samples = generate_samples(f_batch)
            for sa in samples:
                sam = ''.join(sa)
                samp = sam.strip('!')
                if (samp != '') and (not ('!' in samp)) and (sam[0] != '!'):
                    samples_f.append(samp)
                    test_se.append(sa)
                
            gen_length = len(samples_f)                

            if gen_length > 0:
                gen_num += gen_length
                  
                yr_sample_file = open(sample_path + 'sample_' + NAME + '_' + FOLD,'a')
                for samp in samples_f:
                    yr_sample_file.write(samp + '\n')
                yr_sample_file.close()

                test_seq = create_aa_feature(np.asarray([[charmap[ch] for ch in l] for l in test_se]).reshape((gen_length,SEQ_LEN)),gen_length)
                prediction= DLS2F_CNN.predict([test_seq])
                top10_prediction=prediction.argsort()[:,::-1][:,:10]
                
                yr_succ_file = open(sample_path + 'success_sample_' + NAME + '_' + FOLD,'a')                
    
                for p in range(gen_length):
                    f_pre = [fold_index[i] for i in top10_prediction[p]]
                    if FOLD in f_pre:
                        yr_succ_file.write(samples_f[p] + '\n')
                        suc_num += 1

                yr_succ_file.close()

            if gen_num >= GEN_NUM:
                if suc_num >= SUCC_NUM:
                    break
                elif gen_num >= GEN_UPPER:
                    if suc_num == 0:
                        yr_succ_file = open(sample_path + 'success_sample_' + NAME + '_' + FOLD,'a')
                        yr_succ_file.write(FOLD + ': No Successful Sequence.\n')
                        yr_succ_file.close()
                    break
          
        ratio = float(suc_num)/float(gen_num)
        end = time.time()
        run_time = end - start

        yr_fold_file.write(str(ratio) + '\t' + str(suc_num) + '\t' + str(gen_num) + '\t' + str(run_time) + '\n')
        yr_fold_file.close()            
            
            
