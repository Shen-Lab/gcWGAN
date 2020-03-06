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

DATA_DIR = '../../Data/Datasets/Final_Data/'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_language.py!')

BATCH_SIZE = 64 # Batch size
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

model_index = '_'.join(model_index)
result_file = open(result_path + 'loss_' + model_index + '_' + SET_KIND,'w')
result_file.write('Gen_Loss\tCrit_Loss\n')
result_file.close()

seqs_list, folds_list, folds_dict, charmap, inv_charmap = data_helpers.load_dataset_protein_diffset( #MK change
    max_length=SEQ_LEN,
    max_n_examples=MAX_N_EXAMPLES,
    set_kind=SET_KIND,
    data_dir=DATA_DIR
)

fold_unique = []
for info in folds_list:
    if not info in fold_unique:
        fold_unique.append(info)

print 'Fold amount: %d'%len(fold_unique)
print 'Sequence amount: %d'%len(seqs_list)

if len(seqs_list) != len(folds_list):
    print 'Error! Sequences amount is not equal to labels amount!'
    quit()

print ''

lib.print_model_settings(locals().copy())

#seqs, folds, folds_dict, charmap, inv_charmap = data_helpers.load_dataset_protein( #MK change
#    max_length=SEQ_LEN,
#    max_n_examples=MAX_N_EXAMPLES,
#    data_dir=DATA_DIR
#)

#unique_train = DataLoading.file_list(DATA_DIR + 'unique_fold_train') #SZ add
#unique_new = DataLoading.file_list(DATA_DIR + 'unique_fold_new')  #SZ add

print 'Data loading successfully!'

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
disc_cost += LAMBDA*gradient_penalty

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

check_dic = {}
f_list = os.listdir(check_path)
for f in f_list:
    f = f.split('.')[0].split('_')
    if f[0] == 'model':
        check_dic[f[1]] = f[2]

check_index = sorted([int(i) for i in check_dic.keys()])

print 'Load the index of check points successfully!'

# Dataset iterator
def inf_train_gen(seqs_list,folds_list):
    epoch = 0
    while True:
        indices = np.arange(len(seqs_list),dtype=np.int) #MK add
        np.random.shuffle(indices) #MK add
        seqs_list =  [ seqs_list[i] for i in indices] #MK add
        folds_list =  [ folds_list[i] for i in indices] #MK add
        length = len(seqs_list)-BATCH_SIZE+1
        for i in xrange(0, length, BATCH_SIZE):
            yield np.array(    #MK change
                [[charmap[c] for c in l] for l in seqs_list[i:i+BATCH_SIZE]], 
                dtype='int32'
            ),np.array(
                [l for l in folds_list[i:i+BATCH_SIZE]], 
                dtype='float32'
            )

saver  = tf.train.Saver(max_to_keep=None)

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

    gen = inf_train_gen(seqs_list,folds_list)
    
    keys_all = folds_dict.keys() #SZ change 
    l_all = len(keys_all) #SZ change

    for epoch in check_index:
        start_time = time.time()

        print(epoch)

        if epoch > 0 and epoch <= EPOCH_NUM:

            result_file = open(result_path + 'loss_' + model_index + '_' + SET_KIND,'a')
            saver.restore(session,check_path + "model_"+str(epoch)+"_" + check_dic[str(epoch)] + ".ckpt")
            s,f = gen.next() #MK add
            g_cos = session.run(gen_cost,feed_dict={real_inputs_label:f})
            result_file.write(str(g_cos) + '\t')  #SZ add

            _disc_cost = session.run(
                    disc_cost,
                    feed_dict={real_inputs_discrete:s,real_inputs_label:f}
            )

            result_file.write(str(_disc_cost) + '\n')  #SZ add

            result_file.close() 
    
        lib.plot.tick() 
