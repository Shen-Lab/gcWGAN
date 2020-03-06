##################################################################################################
# gcWGAN Training Process.
# Requirement: Weights of the DeepSF model
#              Check points from the semi-supervised learning
##################################################################################################

################################### Load Packages ################################################
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

################################### Set Global Parameters ############################################

Learning_rate = float(sys.argv[1])
CRITIC_ITERS = int(sys.argv[2]) # How many critic iterations per generator iteration. We
                  # use 10 for the results in the paper, but 5 should work fine
                  # as well.
noise_len = int(sys.argv[3])
LAMBDA_g = float(sys.argv[4])

BATCH_SIZE = 64 # Batch size
ITERS = 6000 # How many iterations to train for
SEQ_LEN = 160 # Sequence length in characters
DIM = 512 # Model dimensionality. This is fairly slow and overfits, even on
          # Billion Word. Consider decreasing for smaller datasets.
LAMBDA = 10 # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 50000 # Max number of data examples to load. If data loading
                          # is too slow or takes too much RAM, you can decrease
                          # this (at the expense of having less training data).

fold_len = 20 #MK add
name = '_' + str(Learning_rate) + '_' + str(CRITIC_ITERS) + '_' + str(noise_len) + '_' + str(LAMBDA_g)+'_semi_diff'

################################### Set Paths ####################################################

DATA_DIR = '../Data/Datasets/Final_Data/'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_language.py!')
check_path = '../Checkpoints/gcWGAN/Checkpoints'+name
sample_path = 'gcWGAN_Training_Samples/TrainingSamples'+name
os.system('mkdir ' + check_path)
os.system('mkdir ' + sample_path)
if not os.path.exists('../Checkpoints'):
    os.system('mkdir ../Checkpoints')
if not os.path.exists('../Checkpoints/gcWGAN'):
    os.system('mkdir ../Checkpoints/gcWGAN')
if not os.path.exists(check_path):
    os.system('mkdir ' + check_path)
if not os.path.exists('gcWGAN_Training_Samples'):
    os.system('mkdir gcWGAN_Training_Samples')
if not os.path.exists(sample_path):
    os.system('mkdir ' + sample_path)
lib.print_model_settings(locals().copy())

################################### Load Oracle (DeepSF) ##############################################

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

################################### Load Data ####################################################

seqs, folds, folds_name, folds_dict, charmap, inv_charmap = language_helpers.load_dataset_protein( #MK change
    max_length=SEQ_LEN,
    max_n_examples=MAX_N_EXAMPLES,
    data_dir=DATA_DIR
)

def file_list(path):
    f = open(path,'r')
    lines = f.readlines()
    result = []
    for i in lines:
       line = i.strip('\n')
       result.append(line)
    return result

unique_train = file_list(DATA_DIR + 'unique_fold_train') #SZ add
unique_new = file_list(DATA_DIR + 'unique_fold_new')  #SZ add

print 'Data loading successfully!'

####################### Construct the Structure of the Model ########################################

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

####################### Construct the Loss Functions #########################################

real_inputs_discrete = tf.placeholder(tf.int32, shape=[BATCH_SIZE, SEQ_LEN])
real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
real_inputs_label_deepsf = tf.placeholder(tf.int32, shape=[BATCH_SIZE,])
real_inputs_label = tf.placeholder(tf.float32, shape=[BATCH_SIZE, fold_len]) #MK add
fake_inputs = Generator(BATCH_SIZE,real_inputs_label) #MK change
fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)

fake_inputs_deepsf = tf.nn.softmax(tf.slice(fake_inputs,[0,0,0],[BATCH_SIZE,SEQ_LEN,20]))

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
# previous implementation
#real_inputs_label_deepsf_expand = tf.reshape(tf.tile(tf.reshape(real_inputs_label_deepsf,[BATCH_SIZE,1]),tf.constant([1,topK])),[BATCH_SIZE,topK])
#predicted_y = tf.reduce_sum(tf.cast(tf.equal(real_inputs_label_deepsf_expand,topK_indices),tf.float32),axis=1)

oracle_cost = -tf.reduce_sum(tf.multiply(predicted_y,tf.log(top1))+tf.multiply(predicted_y_inv,tf.log(1-top1)))


disc_real = Discriminator(real_inputs,real_inputs_label) #MK change 
disc_fake = Discriminator(fake_inputs,real_inputs_label) #MK change

disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = -tf.reduce_mean(disc_fake) + LAMBDA_g * oracle_cost


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
disc_cost += LAMBDA*gradient_penalty
G_cost = []  #SZ add
D_cost = []  #SZ add
D_probability = [] #SZ add

gen_params = lib.params_with_name('Generator.')
disc_params = lib.params_with_name('Discriminator.')

####################### Set the Optimizers ##########################################################

gen_train_op = tf.train.AdamOptimizer(learning_rate=Learning_rate, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(learning_rate=Learning_rate, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

####################### Load the Indexes of the Checkpoints #########################################

check_dic = {}
f_list = os.listdir('../Checkpoints/WarmStart/Labeled_Training/')
for f in f_list:
    f = f.split('.')[0].split('_')
    if f[0] == 'model':
        check_dic[f[1]] = f[2]

check_index = sorted([int(i) for i in check_dic.keys()])
c_restore = check_index[99]
r_restore = int(check_dic[str(c_restore)])

print c_restore,r_restore

print 'Load the index of check points successfully!'

###################################### Dataset iterator #############################################

def inf_train_gen(seqs,folds,folds_name):
    epoch = 0 #SZ
    while True:
        indices = np.arange(len(seqs),dtype=np.int) #MK add
        np.random.shuffle(indices) #MK add
        seqs =  [ seqs[i] for i in indices] #MK add
        folds =  [ folds[i] for i in indices] #MK add
        folds_name =  [ folds_name[i] for i in indices] #MK add 
        length = len(seqs)-BATCH_SIZE+1 #SZ add
        for i in xrange(0, length, BATCH_SIZE):
            if i + BATCH_SIZE >= length:  #SZ add
                epoch += 1
 
            yield np.array(    #MK change
                [[charmap[c] for c in l] for l in seqs[i:i+BATCH_SIZE]], 
                dtype='int32'
            ),np.array(
                [l for l in folds[i:i+BATCH_SIZE]], 
                dtype='float32'
            ),np.array(
                [folds_deepsf[l] for l in folds_name[i:i+BATCH_SIZE]],
                dtype='int32'
            ),epoch #SZ

################################### Variables to be Saved ###########################################

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

################################ Begin the Training Process ##########################################

saver  = tf.train.Saver(var_list,max_to_keep=None)

with tf.Session() as session:

    ###### load the check points from the semi-supervised learning process and DeepSF ################

    saver.restore(session,"../Checkpoints/WarmStart/Labeled_Training/model_"+str(c_restore)+"_" + check_dic[str(c_restore)] + ".ckpt")
    DLS2F_ResCNN.load_weights(model_weight)
    DLS2F_ResCNN.trainable = False

    """
    convert theano trained weights to tf: convolution weights should be fliped 
    """

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
    
    ####################### Begin the Iterations ########################################

    ep_before = 0 #SZ add
   
    for iteration in xrange(ITERS):
        start_time = time.time()
        print(iteration)
        ################################ Generator Training #############################################
        if iteration > 0:
            s,f,f_deepsf,epoch = gen.next() #MK add
            g_cos,_ = session.run([gen_cost,gen_train_op],feed_dict={real_inputs_label:f,real_inputs_label_deepsf:f_deepsf,K.learning_phase(): 0}) #MK change
            G_cost.append(g_cos)  #SZ add


        #################################### Critic Training ############################################
        for i in xrange(CRITIC_ITERS):
            s,f,f_deepsf,epoch = gen.next() #MK change
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_inputs_discrete:s,real_inputs_label:f,real_inputs_label_deepsf:f_deepsf,K.learning_phase(): 0}
            )
            if i == CRITIC_ITERS - 1:
                #d_cos = session.run(disc_cost,feed_dict={real_inputs_discrete:s,real_inputs_label:f})  #SZ add
                D_cost.append(_disc_cost)  #SZ add


        lib.plot.plot('time', time.time() - start_time)
        lib.plot.plot('train disc cost', _disc_cost)
         
        ################### Each epoch: save the checkpoints and generatred samples ######################

        if epoch != ep_before:
            ep_before = epoch
            saver.save(session,check_path + "/model_{}".format(epoch) + "_{}.ckpt".format(iteration))

            print "Epoch:", epoch

            keys_train = np.random.choice(unique_train,size=BATCH_SIZE) #MK add
            f_train = [folds_dict[k] for k in keys_train] #MK add
            samples_train = generate_samples(f_train) #MK change

            with open(sample_path +'/train_samples_{}'.format(epoch) + '_{}.txt'.format(iteration), 'w') as f:  #SZ change
                f.write(str(g_cos) + '\t' + str( _disc_cost) + '\n')
                for i in xrange(BATCH_SIZE):
                    s = "fold "+keys_train[i]+": "+''.join(samples_train[i]) #MK change
                    f.write(s + "\n")

            keys_new = np.random.choice(unique_new,size=BATCH_SIZE) #MK add
            f_new = [folds_dict[k] for k in keys_new] #MK add
            samples_new = generate_samples(f_new) #MK change

            with open(sample_path +'/new_samples_{}'.format(epoch) + '_{}.txt'.format(iteration), 'w') as f:  #SZ change
                f.write(str(g_cos) + '\t' + str( _disc_cost) + '\n')
                for i in xrange(BATCH_SIZE):
                    s = "fold "+keys_new[i]+": "+''.join(samples_new[i]) #MK change
                    f.write(s + "\n")

 
        lib.plot.tick()

