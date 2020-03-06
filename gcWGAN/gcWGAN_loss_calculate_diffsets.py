import os, sys
sys.path.append(os.getcwd())

import time

import numpy as np
import tensorflow as tf
import language_helpers
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import tflib.plot

#import Assessment #SZ add
#from Bio.SubsMat import MatrixInfo as matlist #SZ add
import matplotlib.pyplot as plt #SZ add
#matrix = matlist.blosum62 #SZ add

from keras.utils.np_utils import convert_kernel
from keras.models import model_from_json
from keras.engine.topology import Layer
#import theano.tensor as T
from keras import backend as K
from keras.constraints import maxnorm

from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Merge, Convolution1D,Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import MaxPooling1D,GlobalAveragePooling1D

DATA_DIR = '../Data/Datasets/Final_Data/'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_language.py!')

BATCH_SIZE = 64 # Batch size
#ITERS = 6000 # How many iterations to train for
EPOCH_NUM = 100
SEQ_LEN = 160 # Sequence length in characters
DIM = 512 # Model dimensionality. This is fairly slow and overfits, even on
          # Billion Word. Consider decreasing for smaller datasets.
LAMBDA = 10 # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 50000 # Max number of data examples to load. If data loading
                          # is too slow or takes too much RAM, you can decrease
                          # this (at the expense of having less training data).

fold_len = 20 #MK add

check_path = sys.argv[1]
SET_KIND = sys.argv[2]
result_path = sys.argv[3]

if not check_path.endswith('/'):
    check_path += '/'
if not result_path.endswith('/'):
    result_path += '/'

model_index = check_path.strip('/').split('/')[-1]
model_index = model_index.split('_')[1:]

Learning_rate = float(model_index[0])
CRITIC_ITERS = int(model_index[1]) 
noise_len = int(model_index[2])
LAMBDA_g = float(model_index[3])

model_index = '_'.join(model_index)
result_file = open(result_path + 'loss_' + model_index + '_' + SET_KIND,'w')
result_file.write('Gen_Loss\tCrit_Loss\tOverall_Loss\n')
result_file.close()

#sample_path = result_path + 'Loss_samples_' + model_index + '/'
#os.system('mkdir ' + sample_path)

lib.print_model_settings(locals().copy())

###### DeepSF

model_file="./DeepSF_model_weight/model-train-DLS2F.json"
model_weight="./DeepSF_model_weight/model-train-weight-DLS2F.h5"
deepsf_fold="./DeepSF_model_weight/DeepSF_fold.txt"
kmaxnode=30

folds_deepsf = {}
path = deepsf_fold
num_fold=0
count=0
with open(path, 'r') as f:
   for line in f:
       count += 1
       line = line.strip().split()
       if count == 1:
          continue
       key = line[0]
       value = int(line[1])
       folds_deepsf[key] = value
       num_fold += 1

def softargmax(x, beta=1e10):
  x = tf.convert_to_tensor(x)
  x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
  return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)

class K_max_pooling1d(Layer):
    def __init__(self,  ktop, **kwargs):
        self.ktop = ktop
        super(K_max_pooling1d, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],self.ktop,input_shape[2])

    def call(self,x,mask=None):
        n_batch = x.shape[0]
        n_filter = x.shape[2]
        total_len = x.shape[1]
        x = tf.transpose(x,perm = [0,2,1])
        val,ind = tf.nn.top_k(x,k=self.ktop)
        output = tf.transpose(val,perm = [0,2,1])
        
        return output

    def get_config(self):
        config = {'ktop': self.ktop}
        base_config = super(K_max_pooling1d, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def _conv_bn_leakyrelu1D(nb_filter, nb_row, subsample,use_bias=True): #MK added
    def f(input):
        conv = Convolution1D(nb_filter=nb_filter, filter_length=nb_row, subsample_length=subsample,bias=use_bias,
                             init="he_normal", activation='leakyrelu', border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=2)(conv)
        return LeakyReLU(alpha=0.1)(norm)

    return f

def _conv_bn_leakyrelu1D_resnet(nb_filter, nb_row, subsample,use_bias=True): #MK added
    def f(input):
        conv = Convolution1D(nb_filter=nb_filter, filter_length=nb_row, subsample_length=subsample,bias=use_bias,
                             init="he_normal", activation='leakyrelu', border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=2)(conv)

        return Merge(mode='sum')([LeakyReLU(alpha=0.1)(norm),input])

    return f

#### DeepSF

#seqs, folds, folds_name, folds_dict, charmap, inv_charmap = language_helpers.load_dataset_protein( #MK change
#    max_length=SEQ_LEN,
#    max_n_examples=MAX_N_EXAMPLES,
#    data_dir=DATA_DIR
#)

seqs, folds, folds_name, folds_dict, charmap, inv_charmap = language_helpers.load_dataset_protein_diffset( #MK change
    max_length=SEQ_LEN,
    max_n_examples=MAX_N_EXAMPLES,
    set_kind=SET_KIND,
    data_dir=DATA_DIR
)

fold_unique = []
for info in folds:
    if not info in fold_unique:
        fold_unique.append(info)

print 'Fold amount: %d'%len(fold_unique)
print 'Sequence amount: %d'%len(seqs)

if len(seqs) != len(folds):
    print 'Error! Sequences amount is not equal to labels amount!'
    quit()

print ''

print 'Data loading successfully!'

def prepare_seq_deepsf(inp,batch_size,len_feature):
  num = batch_size
  check = tf.zeros([len_feature],tf.int32)
  for i in range(num):
      x = tf.slice(inp,[i,0],[1,len_feature])
      x = tf.cast(tf.reshape(x,[len_feature]),tf.int32)
      s = tf.subtract(tf.constant(len_feature),tf.reduce_sum(tf.sign(x)))
      x = tf.cond(tf.greater_equal(s,tf.constant(len_feature,tf.int32)),lambda:tf.add(x,tf.scatter_nd(tf.constant([[0]]),tf.constant([1]) , tf.constant([len_feature]))),lambda:x)
      index = tf.cond(tf.less_equal(s,tf.constant(0,tf.int32)),lambda:tf.constant(len_feature,tf.int32),lambda:tf.argmax(tf.cast(tf.less_equal(x,tf.constant(0,tf.int32)),tf.int32),output_type=tf.int32))
      x = tf.slice(x,[0],[index])
      x = tf.reshape(x,[1,index])
      x = tf.one_hot(tf.subtract(x,tf.ones([1,index],tf.int32)),20)
      x = tf.cond(tf.less_equal(s,tf.constant(0,tf.int32)),lambda:x,lambda:tf.concat([x,tf.zeros([1,tf.subtract(tf.constant(len_feature),index),20])],axis=1))
      if i==0:
         output = tf.identity(x)
      else:
         output = tf.concat([output,x],axis=0)

  return output


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
real_inputs_label_deepsf = tf.placeholder(tf.int32, shape=[BATCH_SIZE,])
real_inputs_label = tf.placeholder(tf.float32, shape=[BATCH_SIZE, fold_len]) #MK add
fake_inputs = Generator(BATCH_SIZE,real_inputs_label) #MK change
fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)

fake_inputs_deepsf = tf.nn.softmax(tf.slice(fake_inputs,[0,0,0],[BATCH_SIZE,SEQ_LEN,20]))
#########DeepSF

DLS2F_input_shape =(None,20)
filter_sizes=[6,10]
DLS2F_input = Input(tensor=fake_inputs_deepsf)
DLS2F_convs = []
for fsz in filter_sizes:
    DLS2F_conv = DLS2F_input
    for i in range(0,5):
        DLS2F_conv = _conv_bn_leakyrelu1D(nb_filter=40, nb_row=fsz, subsample=1,use_bias=True)(DLS2F_conv)

    for i in range(0,10):
        DLS2F_conv = _conv_bn_leakyrelu1D_resnet(nb_filter=40, nb_row=fsz, subsample=1,use_bias=True)(DLS2F_conv)

    for i in range(0,5):
        DLS2F_conv = _conv_bn_leakyrelu1D(nb_filter=40, nb_row=fsz, subsample=1,use_bias=True)(DLS2F_conv)


    DLS2F_pool = K_max_pooling1d(ktop=kmaxnode)(DLS2F_conv)
    DLS2F_flatten = Reshape((40*kmaxnode,))(DLS2F_pool)
    DLS2F_convs.append(DLS2F_flatten)

if len(filter_sizes)>1:
   DLS2F_out = Merge(mode='concat')(DLS2F_convs)
else:
   DLS2F_out = DLS2F_convs[0]

DLS2F_dense1 = Dense(output_dim=500, init='he_normal', activation='sigmoid', W_constraint=maxnorm(3))(DLS2F_out)
DLS2F_dropout1 = Dropout(0.2)(DLS2F_dense1)
DLS2F_output = Dense(output_dim=1215, init="he_normal", activation="softmax")(DLS2F_dropout1)

DLS2F_ResCNN = Model(input=[DLS2F_input], output=DLS2F_output)

top1 = tf.reduce_max(DLS2F_output,axis=1)-0.01
top10_app = 0.1*top1
ind = tf.concat([tf.reshape(tf.range(BATCH_SIZE),[BATCH_SIZE,1]),tf.reshape(real_inputs_label_deepsf,[BATCH_SIZE,1])],axis=1)
actual_val = tf.gather_nd(DLS2F_output,ind)
predicted_y = tf.maximum(tf.subtract(actual_val,top10_app),0)
predicted_y_inv = 10*tf.minimum(tf.subtract(actual_val,top10_app),0)

oracle_cost = -tf.reduce_sum(tf.multiply(predicted_y,tf.log(top1))+tf.multiply(predicted_y_inv,tf.log(1-top1)))

disc_real = Discriminator(real_inputs,real_inputs_label) #MK change 
disc_fake = Discriminator(fake_inputs,real_inputs_label) #MK change

disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = -tf.reduce_mean(disc_fake) + LAMBDA_g * oracle_cost  # Generator Loss


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
disc_cost += LAMBDA*gradient_penalty  # Critic Loss

overall_cost = disc_cost + LAMBDA_g * oracle_cost  # Overall Loss

gen_params = lib.params_with_name('Generator.')
disc_params = lib.params_with_name('Discriminator.')

check_dic = {}
f_list = os.listdir(check_path)
for f in f_list:
    f = f.split('.')[0].split('_')
    if f[0] == 'model':
        check_dic[f[1]] = f[2]

check_index = sorted([int(i) for i in check_dic.keys()])

print 'Load the index of check points successfully!'

# Dataset iterator
def inf_train_gen(seqs,folds,folds_name):
    while True:
        indices = np.arange(len(seqs),dtype=np.int) #MK add
        np.random.shuffle(indices) #MK add
        seqs =  [ seqs[i] for i in indices] #MK add
        folds =  [ folds[i] for i in indices] #MK add
        folds_name =  [ folds_name[i] for i in indices] #MK add 
        length = len(seqs)-BATCH_SIZE+1 #SZ add
        for i in xrange(0, BATCH_SIZE, BATCH_SIZE):
            yield np.array(    #MK change
                [[charmap[c] for c in l] for l in seqs[i:i+BATCH_SIZE]], 
                dtype='int32'
            ),np.array(
                [l for l in folds[i:i+BATCH_SIZE]], 
                dtype='float32'
            ),np.array(
                [folds_deepsf[l] for l in folds_name[i:i+BATCH_SIZE]],
                dtype='int32'
            )

var_list = []
var_list_others = []
for v in tf.global_variables():
   if "Discriminator." in v.name:
      var_list.append(v)
   if "Generator." in v.name:
      var_list.append(v)
   if "beta1_power" in v.name:
      var_list_others.append(v)
   if "beta2_power" in v.name:
      var_list_others.append(v)

saver  = tf.train.Saver(var_list,max_to_keep=None)

with tf.Session() as session:

    DLS2F_ResCNN.load_weights(model_weight)
    DLS2F_ResCNN.trainable = False
  
    ## convert theano trained weights to tf: convolution weights should be fliped
    ops = []
    for layer in DLS2F_ResCNN.layers:
      if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
        original_w = K.get_value(layer.W)
        converted_w = convert_kernel(original_w)
        ops.append(tf.assign(layer.W, converted_w).op)

    K.get_session().run(ops) 
    session.run(tf.variables_initializer(var_list_others))
     
    print 'initialize weights'

    def generate_samples(label): #MK change
        samples = session.run(fake_inputs,feed_dict={real_inputs_label:label,K.learning_phase(): 0}) #MK change
        samples = np.argmax(samples, axis=2)
        decoded_samples = []
        for i in xrange(len(samples)):
            decoded = []
            for j in xrange(len(samples[i])):
                decoded.append(inv_charmap[samples[i][j]])
            decoded_samples.append(tuple(decoded))
        return decoded_samples

    gen = inf_train_gen(seqs,folds,folds_name)

    keys_all = folds_dict.keys() #SZ change 
    l_all = len(keys_all) #SZ change

    for epoch in check_index:
        start_time = time.time()
        print(epoch)
     
        if epoch > 0 and epoch <= EPOCH_NUM:

            result_file = open(result_path + 'loss_' + model_index + '_' + SET_KIND,'a')
            saver.restore(session,check_path + "model_"+str(epoch)+"_" + check_dic[str(epoch)] + ".ckpt")
            s,f,f_deepsf = gen.next() #MK add
            g_cos = session.run(gen_cost,feed_dict={real_inputs_label:f,real_inputs_label_deepsf:f_deepsf,K.learning_phase(): 0}) #MK change
            result_file.write(str(g_cos) + '\t')  #SZ add

            _disc_cost = session.run(
                    disc_cost,
                    feed_dict={real_inputs_discrete:s,real_inputs_label:f,real_inputs_label_deepsf:f_deepsf,K.learning_phase(): 0}
            )

            result_file.write(str(_disc_cost) + '\t')  #SZ add

            O_cost = session.run(
                    overall_cost,
                    feed_dict={real_inputs_discrete:s,real_inputs_label:f,real_inputs_label_deepsf:f_deepsf,K.learning_phase(): 0}
            )
            
            result_file.write(str(O_cost) + '\n')
            
            result_file.close()

        lib.plot.plot('time', time.time() - start_time)
        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.tick()
