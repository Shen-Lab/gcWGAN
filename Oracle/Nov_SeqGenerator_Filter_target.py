################################ Import packages #################################

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

import Oracle_Filters # Contain oracle filters

############################## Set paths and parameters ################################

Checkpoint = sys.argv[1]    # Path of the check point
while Checkpoint.endswith('.'):
    Checkpoint = Checkpoint[:-1]
print Checkpoint 
#FOLD = sys.argv[2]          # Name of the fold 
KIND = sys.argv[2]          # Set the terminate criteria, "All" or "Success"
GEN_NUM = int(sys.argv[3])  # Terminate amount
TARGET = sys.argv[4]

target_name = '_'.join(TARGET.split('.'))


if GEN_NUM <= 0:
    print 'Error! The number of generated sequences should be positive.'
    quit()

All_Flag = (KIND.upper() == 'ALL')
Suc_Flag = (KIND.upper() == 'SUCCESS')
if not (Suc_Flag or All_Flag):
    print 'Error! No kind named %s'%KIND
    quit()

if All_Flag:
    All_Num = GEN_NUM
    Suc_Num = None
if Suc_Flag:
    All_Num = None
    Suc_Num = GEN_NUM 

generated_file_index = sys.argv[5]
JOB_INDEX = sys.argv[6]

if len(sys.argv) >= 8 and not ('DeepSF' in sys.argv[7]):
    MIN_LEN = int(sys.argv[7])
else:
    MIN_LEN = 60
if len(sys.argv) >= 9 and not ('DeepSF' in sys.argv[8]):
    MAX_LEN = int(sys.argv[8])
else:
    MAX_LEN = 160

if 'M_DeepSF' in sys.argv:
    M_DeepSF_flag = True
else: 
    M_DeepSF_flag = False
if 'O_DeepSF' in sys.argv:
    O_DeepSF_flag = True
else:
    O_DeepSF_flag = False

if not(M_DeepSF_flag or O_DeepSF_flag) and Suc_Flag:
    print 'Error! No filter input!'
    quit()

#fold_name = FOLD.split('.')[0] + '_' + FOLD.split('.')[1]
fold_name = 'nov'

noise_len = 64

DATA_DIR = '../Data/Datasets/Final_Data/'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_language.py!')

BATCH_SIZE = 200 # Batch size
SEQ_LEN = 160 # Sequence length in characters
DIM = 512 # Model dimensionality. This is fairly slow and overfits, even on
          # Billion Word. Consider decreasing for smaller datasets.
LAMBDA = 10 # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 50000 # Max number of data examples to load. If data loading
                          # is too slow or takes too much RAM, you can decrease
                          # this (at the expense of having less training data).
TOP_NUM = 10   # Oracle condition

fold_len = 20 #MK add

lib.print_model_settings(locals().copy())

################################ Load Data #################################

seqs, folds, folds_dict, charmap, inv_charmap = data_helpers.load_dataset_protein( #MK change
    max_length=SEQ_LEN,
    max_n_examples=MAX_N_EXAMPLES,
    data_dir=DATA_DIR
)

novel_array = np.loadtxt(DATA_DIR + 'novel_coordinate')
fold_vector = list(novel_array)

print 'Data loading successfully!'

########################### Modified DeepSF #######################

if M_DeepSF_flag:
    """
    Activate Modified DeepSF 
    """
    ###### Set the path of the model parameters and indexes ######
    
    M_DeepSF_path = 'DeepSF_modified/DeepSF_model_weight_more_folds/'
    M_model_file = M_DeepSF_path + "model-train-DLS2F.json"
    M_model_weight = M_DeepSF_path + "model-train-weight-DLS2F.h5"
    M_deepsf_fold = M_DeepSF_path + "fold_label_relation2.txt"
    M_fold_index = DataLoading.Accuracy_index(path = M_DeepSF_path +'fold_label_relation2.txt')
    
    ###### Load the models ######
    
    M_json_file_model = open(M_model_file, 'r')
    M_loaded_model_json = M_json_file_model.read()
    M_json_file_model.close()
    M_DLS2F_CNN = model_from_json(M_loaded_model_json, custom_objects={'K_max_pooling1d': Oracle_Filters.K_max_pooling1d})
    M_DLS2F_CNN.load_weights(M_model_weight)
    M_DLS2F_CNN.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer="nadam")

    print 'Modified DeepSF loading successfully!'

################################ Original DeepSF ##################

if O_DeepSF_flag:
    """
    Activate Modified DeepSF 
    """   
   
    ###### Set the global parameters ######

    SCRATCH_PATH = 'DeepSF_origin/DeepSF/software/SCRATCH-1D_1.1/bin/run_SCRATCH-1D_predictors.sh'
    PSSM_database = 'DeepSF_origin/PSSM/nr90/nr90'
    pgp_path = 'DeepSF_origin/PSSM/blast-2.2.26/bin/blastpgp'

    AA_dict = {'A':0,'R':1,'N':2,'D':3,'C':4,'Q':5,'E':6,'G':7,'H':8,'I':9,'L':10,'K':11,'M':12,'F':13,'P':14,'S':15,'T':16,'W':17,'Y':18,'V':19}
    SS_dict = {'C':0,'E':1,'H':2}
    SA_dict = {'e':0,'-':1}
 
    ###### Set the path of the temperory files ######
    
    O_Seq_Stat_file = open(generated_file_index + '_Origin_All_Stat' + '_target_' + target_name,'w')
    O_Seq_Stat_file.close()

    folder_index = 0
    while os.path.exists('Temporary_' + fold_name + '_' + str(JOB_INDEX) + '_' + str(folder_index)):
        folder_index += 1
    Temp_path = 'Temporary_' + fold_name + '_' + str(JOB_INDEX) + '_' + str(folder_index)
    os.system('mkdir ' + Temp_path)
    os.system('mkdir ' + Temp_path + '/Seq')
    os.system('mkdir ' + Temp_path + '/PSSM')
    os.system('mkdir ' + Temp_path + '/AA_SS_SA')

    ###### Set the path of the model parameters and indexes ######
    
    kmaxnode=30
    min_pssm=-8
    max_pssm=16

    O_DeepSF_path = 'DeepSF_origin/'
    O_DeepSF_model_path = 'DeepSF_origin/DeepSF/' 
    O_relationfile = O_DeepSF_model_path + 'datasets/D1_SimilarityReduction_dataset/fold_label_relation2.txt'
    O_model_file = O_DeepSF_model_path + 'models/model_SimilarityReduction.json'    # path of the model file
    O_model_weight = O_DeepSF_model_path + 'models/model_SimilarityReduction.h5'    # path of the weight file    
    
    O_label2fold = dict()
    O_rela_file=open(O_relationfile,'r').readlines()
    for i in xrange(len(O_rela_file)):
        if O_rela_file[i].find('Label') >0 :
            print "Skip line ",O_rela_file[i]
            continue
        fold = O_rela_file[i].rstrip().split('\t')[0]
        label = O_rela_file[i].rstrip().split('\t')[1]
        if label not in O_label2fold:
            O_label2fold[label]=fold

    ###### Load the models ######

    O_json_file_model = open(O_model_file, 'r')
    O_loaded_model_json = O_json_file_model.read()
    O_json_file_model.close()
    O_DLS2F_CNN = model_from_json(O_loaded_model_json, custom_objects={'K_max_pooling1d': Oracle_Filters.K_max_pooling1d})
    O_DLS2F_CNN.load_weights(O_model_weight)
    O_DLS2F_CNN.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer="nadam")
    O_get_flatten_layer_output = K.function([O_DLS2F_CNN.layers[0].input, K.learning_phase()],[O_DLS2F_CNN.layers[-3].output]) # input to flatten layer
 
################################ Structure of the model ##############

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

################################ Generate sequences #################################

saver  = tf.train.Saver()

with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    def generate_samples(label):
        samples = session.run(fake_inputs,feed_dict={real_inputs_label:label})
        samples = np.argmax(samples, axis=2)
        decoded_samples = []
        for i in xrange(len(samples)):
            decoded = []
            for j in xrange(len(samples[i])):
                decoded.append(inv_charmap[samples[i][j]])
            decoded_samples.append(tuple(decoded))
        return decoded_samples

    saver.restore(session,Checkpoint)

    print 'Restore Successfully!'

    all_num = 0
    suc_num = '-'
    seq_file = open(generated_file_index + '_GenSeq' + '_target_' + target_name,'w')
    seq_file.close()
    time_file = open(generated_file_index + '_time_stat' + '_target_' + target_name,'w')
    time_file.write('valid_num\tsuc_num\ttime\n')
    time_file.close()

    if M_DeepSF_flag or O_DeepSF_flag:
        Stat_file = open(generated_file_index + '_SeqStatus' + '_target_' + target_name,'w')
        Stat_file.write('M_DeepSF\tO_DeepSF\n')
        Stat_file.close()
        Suc_file = open(generated_file_index + '_SucSeq' + '_target_' + target_name,'w')
        Suc_file.close()
   
    start_time = time.time()

    while((all_num < All_Num) or Suc_Flag):

        batch_start_time = time.time()

        ########## Generate sequences ##########

        #f_batch = [folds_dict[FOLD]] * BATCH_SIZE
        f_batch = [fold_vector] * BATCH_SIZE

        #f_batch = []
        #for b_i in range(BATCH_SIZE):
        #    f_batch.append(fold_vector)

        samples = generate_samples(f_batch)
        
        ########## Select valid sequences ##########
        
        samples_f = []
        test_se = []

        for sa in samples:
            sam = ''.join(sa)
            samp = sam.strip('!')
            if ((len(samp) >= MIN_LEN) and (len(samp) <= MAX_LEN)) and ((not ('!' in samp)) and sam[0] != '!'):
                all_num += 1
                test_se.append(sa)    
                samples_f.append(samp)
                with open(generated_file_index + '_GenSeq' + '_target_' + target_name,'a') as seq_file: 
                    seq_file.write(samp + '\n')
                if (all_num >= All_Num) and All_Flag:
                    break
   
        V_SIZE = len(test_se)   
        
        ######################## Filters ##################### 
           
        if V_SIZE > 0 and (M_DeepSF_flag or O_DeepSF_flag):
            
            if suc_num == '-':
                suc_num = 0

            seq_status = []
            success_index = range(V_SIZE)

        ############## Activate the modified DeepSF ##########

            if M_DeepSF_flag: 
                
                M_suc_num = 0
                M_success_index = []
       
                test_seq = Oracle_Filters.create_aa_feature(np.asarray([[charmap[c] for c in l] for l in test_se]).reshape((V_SIZE,SEQ_LEN)),V_SIZE,SEQ_LEN)
                prediction= M_DLS2F_CNN.predict([test_seq])
                top_prediction=prediction.argsort()[:,::-1][:,:TOP_NUM]
                
                for p in range(V_SIZE):
                    f_pre = [M_fold_index[i] for i in top_prediction[p]]
                     
                    if TARGET in f_pre:
                        seq_status.append(['pass','-'])
                        M_success_index.append(p)
                        M_suc_num += 1
                    else: 
                        seq_status.append(['fail','-'])

                success_index = M_success_index[:]

            else:
                #seq_status = [['-']]*V_SIZE
                for p in range(V_SIZE):
                    seq_status.append(['-','-'])
        ############## Activate the modified DeepSF ##########

            if O_DeepSF_flag:
                
                seq_index = 0
                O_suc_num = 0
                O_success_index = []

                for p in success_index:  
                    
                    seq = samples_f[p]
                    seq_index += 1                    
                    seq_name = fold_name + '_' + str(seq_index)
                    fasta_name = seq_name + '.fasta'
 
                    ### Create Fasta sequence files ###
                    
                    Oracle_Filters.Seq_Fasta(seq,Temp_path + '/Seq/' + fasta_name)

                    ### Create the features ###

                    #Oracle_Filters.PSSM(Temp_path + '/Seq/' + fasta_name,PSSM_database,Temp_path + '/PSSM/')
                    Oracle_Filters.PSSM_pgp(Temp_path + '/Seq/' + fasta_name,PSSM_database,Temp_path + '/PSSM/',pgp_path)
                    Oracle_Filters.AA_SS_SA(Temp_path + '/Seq/' + fasta_name,SCRATCH_PATH,Temp_path + '/AA_SS_SA/',AA_dict,SS_dict,SA_dict)
                     
                    ### feature process ###
                      
                    featurefile = Temp_path + '/AA_SS_SA/' + seq_name + '.fea_aa_ss_sa'
                    pssmfile = Temp_path + '/PSSM/' + seq_name + '.pssm_fea'
                    
                    featuredata = Oracle_Filters.import_DLS2FSVM(featurefile)
                    pssmdata = Oracle_Filters.import_DLS2FSVM(pssmfile)
                    pssm_fea = pssmdata[:,1:]

                    fea_len = (featuredata.shape[1]-1)/(20+3+2)
                    train_feature = featuredata[:,1:]
                    train_feature_seq = train_feature.reshape(fea_len,25)
                    train_feature_aa = train_feature_seq[:,0:20]
                    train_feature_ss = train_feature_seq[:,20:23]
                    train_feature_sa = train_feature_seq[:,23:25]
                    train_feature_pssm = pssm_fea.reshape(fea_len,20)

                    train_feature_pssm_normalize = np.empty_like(train_feature_pssm)
                    train_feature_pssm_normalize[:] = train_feature_pssm
                    train_feature_pssm_normalize=(train_feature_pssm_normalize-min_pssm)/(max_pssm-min_pssm)
                    featuredata_all_tmp = np.concatenate((train_feature_aa,train_feature_ss,train_feature_sa,train_feature_pssm_normalize), axis=1)

                    if fea_len <kmaxnode: # suppose k-max = 30
                        fea_len = kmaxnode
                        train_featuredata_all = np.zeros((kmaxnode,featuredata_all_tmp.shape[1]))         
                        train_featuredata_all[:featuredata_all_tmp.shape[0],:featuredata_all_tmp.shape[1]] = featuredata_all_tmp
                    else:
                        train_featuredata_all = featuredata_all_tmp

                    train_featuredata_all=train_featuredata_all.reshape(1,train_featuredata_all.shape[0],train_featuredata_all.shape[1])
                     
                    ### Make prediction and select the sequences ###

                    predict_val= O_DLS2F_CNN.predict([train_featuredata_all])
                    top_prediction=predict_val.argsort()[:,::-1][:,:TOP_NUM]
                    predict_labels = [O_label2fold[str(i)] for i in top_prediction[0]]
                   
                    O_stat_time = time.time() - start_time
                    
                    O_Seq_Stat_file = open(generated_file_index + '_Origin_All_Stat' + '_target_' + target_name,'a')
                    O_Seq_Stat_file.write(seq + '\n')
                    O_Seq_Stat_file.write(str(predict_labels) + '\n')
                     
                    if TARGET in predict_labels:
                        O_success_index.append(p)
                        O_suc_num += 1
                        seq_status[p][1] = 'pass'
                        O_Seq_Stat_file.write('Success' + '\n')
                    else:
                        seq_status[p][1] = 'fail'
                        O_Seq_Stat_file.write('Fail' + '\n')                    
 
                    O_Seq_Stat_file.write(str(O_stat_time) + '\n')
                    O_Seq_Stat_file.write('\n')
                    O_Seq_Stat_file.close()                    

                    ### Delete the temperary files ###        
                    
                    os.system('rm ' + Temp_path + '/Seq/' + seq_name + '.fasta') 
            
                    if (suc_num + O_suc_num >= Suc_Num) and Suc_Flag:
                        break

                success_index = O_success_index[:]
                suc_num += O_suc_num
              
            else:
                suc_num += M_suc_num
                #for status in seq_status:
                #    if len(status) == 1:
                #        status.append('-')

        ### Record the sequence status and the successful sequences ###
            
            samples_f = [samples_f[ind] for ind in success_index]
           
            if len(seq_status) != V_SIZE:
                print 'Error! The sequense set and the status set do not match!'
 
            Stat_file = open(generated_file_index + '_SeqStatus' + '_target_' + target_name,'a')
            for p in range(V_SIZE):
                Stat_file.write(seq_status[p][0] + '\t')
                Stat_file.write(seq_status[p][1] + '\n')
            Stat_file.close()

            Suc_file = open(generated_file_index + '_SucSeq' + '_target_' + target_name,'a')
            for s in samples_f:
                Suc_file.write(s + '\n')
            Suc_file.close()

        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time    

        with open(generated_file_index + '_time_stat' + '_target_' + target_name,'a') as time_file:
            time_file.write(str(all_num) + '\t')
            time_file.write(str(suc_num) + '\t')
            time_file.write(str(batch_time) + '\n')
         
        if (all_num >= All_Num) and All_Flag:
            break
        if (suc_num >= Suc_Num) and Suc_Flag:
            break
             
########################### Remove the Temperary Folder #############################

if O_DeepSF_flag:
    os.system('rm -r ' + Temp_path)


