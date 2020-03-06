###########################################################################################
# Modified from "DLS2F_predict_fea.py" in the DeepSF repository. Predict the target fold of 
# an input sequence.
# 11/15/2019
# Input: sequence
# Output: the sorted fold list of the prediction
###########################################################################################

import sys
import numpy as np
import os
from keras.models import model_from_json
from keras.engine.topology import Layer
import theano.tensor as T
from keras import backend as K
from keras.constraints import maxnorm

from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Merge, Convolution1D
from keras.layers.normalization import BatchNormalization


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

if __name__ == '__main__':

    #print len(sys.argv)
    if len(sys.argv) != 6:
            print 'please input the right parameters: list, model, weight, kmax'
            sys.exit(1)
    
    ####################################### Define the paths and parameters #################################

    DeepSF_path = '../DeepSF/' # path of the deepsf repo    

    fold = sys.argv[1] # fold name
    fold_name = '_'.join(fold.split('.'))
    model_file = DeepSF_path + 'models/model_SimilarityReduction.json'    # path of the model file
    model_weight = DeepSF_path + 'models/model_SimilarityReduction.h5'    # path of the weight file
    Seq_path = sys.argv[2] # path of the sequence file
    feature_dir=sys.argv[3].strip('/') # path of the aa_ss_sa_features
    pssm_dir=sys.argv[4].strip('/')  # path of the PSSM
    resultpath=sys.argv[5].strip('/') # path of the folders that contains the sequences that pass the filter 
    kmaxnode=30
     
    num_all = 0
    success_num = 0

    if not os.path.exists(model_file):
         raise Exception("model file %s not exists!" % model_file)
    if not os.path.exists(model_weight):
         raise Exception("model file %s not exists!" % model_weight)
    
    ###################################### Load the relation file ###########################################
    
    relationfile=DeepSF_path + 'datasets/D1_SimilarityReduction_dataset/fold_label_relation2.txt'
    
    #fold2label = dict()
    label2fold = dict()
    rela_file=open(relationfile,'r').readlines()
    for i in xrange(len(rela_file)):
        if rela_file[i].find('Label') >0 :
            print "Skip line ",rela_file[i]
            continue
        fold = rela_file[i].rstrip().split('\t')[0]
        label = rela_file[i].rstrip().split('\t')[1]
        #if fold not in fold2label:
        #    fold2label[fold]=int(label)
        if label not in label2fold:
            label2fold[label]=fold       
    print 'Size of relation file:',len(label2fold.keys())
    ####################################### Load the model ##################################################
    
    print "Loading Model file ",model_file
    print "Loading Model weight ",model_weight

    ### Load the model

    json_file_model = open(model_file, 'r')
    loaded_model_json = json_file_model.read()
    json_file_model.close()    
    DLS2F_CNN = model_from_json(loaded_model_json, custom_objects={'K_max_pooling1d': K_max_pooling1d})        
    
    print "######## Loading existing weights ",model_weight;
    
    ### Load the weights

    DLS2F_CNN.load_weights(model_weight)
    DLS2F_CNN.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer="nadam")
    get_flatten_layer_output = K.function([DLS2F_CNN.layers[0].input, K.learning_phase()],[DLS2F_CNN.layers[-3].output]) # input to flatten layer
    
    ####################################### Load the data ###################################################

    print "Start loading data"
    Testlist_data_keys = dict()
    Testlist_targets_keys = dict()
    sequence_file=open(Seq_path,'r').readlines() 
    for i in xrange(len(sequence_file)):
        line = sequence_file[i].strip('\n')
        if '>' in line:
            num_all += 1
            seq_name = line[1:]

            j = i + 1 
            seq = ''
            while(not '>' in sequence_file[j]):
                seq += sequence_file[j].strip('\n')
                j += 1
  
            featurefile = feature_dir + '/' + seq_name + '.fea_aa_ss_sa'
            pssmfile = pssm_dir + '/' + seq_name + '.pssm_fea'

            if not os.path.isfile(featurefile):
                print "feature file not exists: ",featurefile, " pass!"
        
            if not os.path.isfile(pssmfile):
                print "pssm feature file not exists: ",pssmfile, " pass!"
        
            featuredata = import_DLS2FSVM(featurefile)
            pssmdata = import_DLS2FSVM(pssmfile)
            pssm_fea = pssmdata[:,1:]
        
            fea_len = (featuredata.shape[1]-1)/(20+3+2)
            train_labels = featuredata[:,0]
            train_feature = featuredata[:,1:]
            train_feature_seq = train_feature.reshape(fea_len,25)
            train_feature_aa = train_feature_seq[:,0:20]
            train_feature_ss = train_feature_seq[:,20:23]
            train_feature_sa = train_feature_seq[:,23:25]
            train_feature_pssm = pssm_fea.reshape(fea_len,20)
            min_pssm=-8
            max_pssm=16
        
            ### Data Process

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
        
            train_targets = np.zeros((train_labels.shape[0], 1195 ), dtype=int)
            for i in range(0, train_labels.shape[0]):
                train_targets[i][int(train_labels[i])] = 1
        
            train_featuredata_all=train_featuredata_all.reshape(1,train_featuredata_all.shape[0],train_featuredata_all.shape[1])
        
            predict_val= DLS2F_CNN.predict([train_featuredata_all])
            #hidden_feature= get_flatten_layer_output([train_featuredata_all,1])[0] ## output in train mode = 1 https://keras.io/getting-started/faq/
        
            top10_prediction=predict_val.argsort()[:,::-1][:,:10]
            
            labels_predict = [label2fold[str(i)] for i in top10_prediction[0]]

            if fold in labels_predict:
                success_num += 1
                with open(resultpath + '/' + fold_name + '_' + str(success_num) + '.fasta','w') as suc_seq_file:
                    suc_seq_file.write('>' + seq_name + '\n')
                    while len(seq) > 60:
                        suc_seq_file.write(seq[0:60] + '\n')
                        seq = seq[60:]
                    suc_seq_file.write(seq + '\n')
        
print "%d out of %d sequences pass the filter."%(success_num,num_all)    
        


