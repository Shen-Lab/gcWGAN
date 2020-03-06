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
from Bio.SubsMat import MatrixInfo as matlist #SZ add
import matplotlib.pyplot as plt #SZ add
matrix = matlist.blosum62 #SZ add

# Download Google Billion Word at http://www.statmt.org/lm-benchmark/ and
# fill in the path to the extracted files here!
DATA_DIR = '../Data/Datasets/Final_Data/'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_language.py!')

"""
SZ change. Set parameters as input arguments.
"""

name = sys.argv[1]

check_path = '../Checkpoints/cWGAN/model_'+name
sample_path = 'cWGAN_Training_Samples/TrainingSamples_'+name

if not os.path.exists('../Checkpoints'):
    os.system('mkdir ../Checkpoints')
if not os.path.exists('../Checkpoints/cWGAN'):
    os.system('mkdir ../Checkpoints/cWGAN')
if not os.path.exists(check_path):
    os.system('mkdir ' + check_path)
if not os.path.exists('cWGAN_Training_Samples'):
    os.system('mkdir cWGAN_Training_Samples')
if not os.path.exists(sample_path):
    os.system('mkdir ' + sample_path)

INDEX = name.split('_')
Learning_rate = float(INDEX[0])
CRITIC_ITERS = int(INDEX[1]) # How many critic iterations per generator iteration. We
                  # use 10 for the results in the paper, but 5 should work fine
                  # as well.
noise_len = int(INDEX[2])

BATCH_SIZE = 64 # Batch size
ITERS = 6000 # How many iterations to train for#
SEQ_LEN = 160 # Sequence length in characters
DIM = 512 # Model dimensionality. This is fairly slow and overfits, even on
          # Billion Word. Consider decreasing for smaller datasets.
LAMBDA = 10 # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 50000 # Max number of data examples to load. If data loading
                          # is too slow or takes too much RAM, you can decrease
                          # this (at the expense of having less training data).

fold_len = 20 #MK add

lib.print_model_settings(locals().copy())

seqs, folds, folds_dict, charmap, inv_charmap = data_helpers.load_dataset_protein( #MK change
    max_length=SEQ_LEN,
    max_n_examples=MAX_N_EXAMPLES,
    data_dir=DATA_DIR
)

inter_dic = DataLoading.Interval_dic(DATA_DIR + 'Interval_1.fa') #SZ add
unique_train = DataLoading.file_list(DATA_DIR + 'unique_fold_train') #SZ add
unique_new = DataLoading.file_list(DATA_DIR + 'unique_fold_new')  #SZ add

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
G_cost = []  #SZ add
D_cost = []  #SZ add
D_probability = [] #SZ add

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

gen_train_op = tf.train.AdamOptimizer(learning_rate=Learning_rate, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(learning_rate=Learning_rate, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

check_dic = {}
f_list = os.listdir(check_path)
for f in f_list:
    f = f.split('.')[0].split('_')
    if f[0] == 'model':
        check_dic[f[1]] = f[2]

check_index = sorted([int(i) for i in check_dic.keys()])

c_restore = check_index[-1]
r_restore = int(check_dic[str(c_restore)])

print 'Load the index of check points successfully!'

# Dataset iterator
def inf_train_gen(seqs,folds):
    epoch = c_restore
    while True:
        indices = np.arange(len(seqs),dtype=np.int) #MK add
        np.random.shuffle(indices) #MK add
        seqs =  [ seqs[i] for i in indices] #MK add
        folds =  [ folds[i] for i in indices] #MK add
        length = len(seqs)-BATCH_SIZE+1
        for i in xrange(0, length, BATCH_SIZE):
            if i + BATCH_SIZE >= length:  #SZ add
                epoch += 1
            yield np.array(    #MK change
                [[charmap[c] for c in l] for l in seqs[i:i+BATCH_SIZE]], 
                dtype='int32'
            ),np.array(
                [l for l in folds[i:i+BATCH_SIZE]], 
                dtype='float32'
            ),epoch #SZ change

saver  = tf.train.Saver(max_to_keep=None)

with tf.Session() as session:

    session.run(tf.initialize_all_variables())
    
    saver.restore(session,check_path + "/model_"+str(c_restore)+"_" + check_dic[str(c_restore)] + ".ckpt")

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

    gen = inf_train_gen(seqs,folds)
    
    keys_all = folds_dict.keys() #SZ change 
    l_all = len(keys_all) #SZ change
        
    ep_before = c_restore #SZ add
    
    for iteration in xrange(r_restore + 1,r_restore + 1 + ITERS):
        start_time = time.time()

        # Train generator
        if iteration > r_restore:
            s,f,epoch = gen.next() #MK add #SZ change
            _ = session.run(gen_train_op,feed_dict={real_inputs_label:f}) #MK change
            g_cos = session.run(gen_cost,feed_dict={real_inputs_label:f}) #SZ add
            #disc_f = session.run(disc_fake,feed_dict={real_inputs_label:f}) #SZ add
            G_cost.append(g_cos)  #SZ add
            #D_probability.append(disc_f) #SZ add

        # Train critic
        for i in xrange(CRITIC_ITERS):
            s,f,epoch = gen.next() #MK change
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_inputs_discrete:s,real_inputs_label:f}
            )
            if i == CRITIC_ITERS - 1:
                #d_cos = session.run(disc_cost,feed_dict={real_inputs_discrete:s,real_inputs_label:f})  #SZ add
                D_cost.append(_disc_cost)  #SZ add
            
        lib.plot.plot('time', time.time() - start_time)
        lib.plot.plot('train disc cost', _disc_cost)
        
        #SZ change
        if epoch > ep_before:
            ep_before = epoch
            saver.save(session,check_path+"/model_{}".format(epoch) + "_{}.ckpt".format(iteration))
            
            print "Epoch:", epoch
    
            keys_train = np.random.choice(unique_train,size=BATCH_SIZE) #MK add
            f_train = [folds_dict[k] for k in keys_train] #MK add
            samples_train = generate_samples(f_train) #MK change
            
            file_c_cost = open(sample_path + '/Critic_Cost','a')
            file_g_cost = open(sample_path + '/Generator_Cost','a')
            file_g_cost.write(str(g_cos) + '\n')
            file_c_cost.write(str(_disc_cost) + '\n')
            file_c_cost.close()
            file_g_cost.close()
 
            with open(sample_path+'/train_samples_{}'.format(epoch) + '_{}.txt'.format(iteration), 'w') as f:  #SZ change
                for i in xrange(BATCH_SIZE):
                    s = "fold "+keys_train[i]+": "+''.join(samples_train[i]) #MK change
                    f.write(s + "\n")
                    
            keys_new = np.random.choice(unique_new,size=BATCH_SIZE) #MK add
            f_new = [folds_dict[k] for k in keys_new] #MK add
            samples_new = generate_samples(f_new) #MK change
            
            with open(sample_path+'/new_samples_{}'.format(epoch) + '_{}.txt'.format(iteration), 'w') as f:  #SZ change
                for i in xrange(BATCH_SIZE):
                    s = "fold "+keys_new[i]+": "+''.join(samples_new[i]) #MK change
                    f.write(s + "\n")

        lib.plot.tick() 
"""
plt.figure(1) #SZ add
plt.plot(range(c_restore,c_restore + len(D_cost)),D_cost) #SZ add
plt.title("Critic Loss Function")
plt.xlabel("Iteration Number")
plt.ylabel("Critic Loss")
plt.savefig('Images/Critic_Loss_'+name+'_after'+str(c_restore)+'.png')
plt.figure(2) #SZ add
plt.plot(range(c_restore,c_restore + len(G_cost)),G_cost) #SZ add
plt.title("Generator Loss Function")
plt.xlabel("Iteration Number")
plt.ylabel("Generator Loss")
plt.savefig('Images/Generator_Loss_'+name+'_after'+str(c_restore)+'.png')
plt.show() #SZ add
"""
