import numpy as np
import os
from numpy import linalg

def TM_score(pdb_1,pdb_2):
    command_1 = './TMalign ' + pdb_1 + ' ' + pdb_2 + ' -a'
    output_1 = os.popen(command_1)
    out_1 = output_1.read()
    if "(if normalized by length of Chain_2" in out_1:
        k_1 = out_1.index("(if normalized by length of Chain_2")
        tms_1 = out_1[k_1-8:k_1-1]
    else:
        return None
    
    command_2 = './TMalign ' + pdb_2 + ' ' + pdb_1 + ' -a'
    output_2 = os.popen(command_2)
    out_2 = output_2.read()
    if "(if normalized by length of Chain_2" in out_2:
        k_2 = out_2.index("(if normalized by length of Chain_2")
        tms_2 = out_2[k_2-8:k_2-1]
    else:
        return None
    return (float(tms_1) + float(tms_2))/2

def TM_1200(pdb,path):
    l = len(os.listdir(path))
    result = []
    for i in range(1,l+1):
        tms = TM_score(pdb,path + str(i) + '.pdb')
        result.append(tms)
    return result

def coordinate_calculation(pdb):
    class eigen:
        def __init__(self):
                self.x=0
        def center(self, X, tm):

                m,n = len(X), len(X[0])

                H1= np.ones((n,n))/float(n)
                H2= np.ones((m,n))/float(n)

                return X - np.matmul(X, H1) - np.matmul(H2,tm) + np.matmul(np.matmul(H2,tm), H1)

        def fit(self, X):
                w, v = linalg.eig(X)

                v=np.transpose(v)

                for i in range(len(w)):
                        for j in range(i+1, len(w)):
                                if(w[i]<w[j]):
                                        temp=w[i]
                                        w[i]=w[j]
                                        w[j]=temp
                                        v[[i,j]]=v[[j,i]]
                return w, v

    tm = np.load("../Datasets/Intermediate_Data/TM_matrix_2_pdbs-noTER_symmetric.npy")
    tm = tm + np.identity(len(tm))*5.6
    tm_test = TM_1200(pdb, '../Datasets/Final_Data/pdbs/')

    model=eigen()
    tm_test_new = model.center(np.array([tm_test]), tm)
    fold_basis=np.loadtxt("../Datasets/Final_Data/folds_basis")

    new_coor=[]

    for i in range(1):
        coor_temp=[]
        for j in range(20):
                coor_temp.append(np.dot(tm_test_new[i],fold_basis[j]))

        new_coor.append(coor_temp)
    return np.array(new_coor)

def TM_score_ref(pdb,ref):
    command_1 = './TMalign ' + pdb + ' ' + ref + ' -a'
    output_1 = os.popen(command_1)
    out_1 = output_1.read()
    if "(if normalized by length of Chain_2" in out_1:
        k_1 = out_1.index("(if normalized by length of Chain_2")
        tms_1 = out_1[k_1-8:k_1-1]
    else:
        return None

    return float(tms_1)

def TM_1200_ref(path,pdb):
    l = len(os.listdir(path))
    result = []
    for i in range(1,l+1):
        tms = TM_score_ref(pdb,path + str(i) + '.pdb')
        result.append(tms)
    return result

def TM_score_fast(pdb_1,pdb_2):
    command = './TMalign ' + pdb_1 + ' ' + pdb_2 + ' -a'
    output = os.popen(command)
    out = output.read()
    if "(if normalized by length of Chain_1" in out:
        k_1 = out.index("(if normalized by length of Chain_1")
        tms_1 = out[k_1-8:k_1-1]
    else:
        return None
    if "(if normalized by length of Chain_2" in out:
        k_2 = out.index("(if normalized by length of Chain_2")
        tms_2 = out[k_2-8:k_2-1]
    else:
        return None
    return (float(tms_1) + float(tms_2))/2

def TM_1200_fast(pdb,path):
    l = len(os.listdir(path))
    result = []
    for i in range(1,l+1):
        tms = TM_score_fast(pdb,path + str(i) + '.pdb')
        result.append(tms)
    return result

def TM_score_all(pdb_1,pdb_2,file_path=None):
    command = './TMalign ' + pdb_1 + ' ' + pdb_2 
    output = os.popen(command)
    out = output.read()
    ### Save output
    if file_path != None:
        if os.path.exists(file_path):
            fil = open(file_path,'a')
            fil.write(out)
        else:
            if not '/' in file_path:
                fil = open(file_path,'w')
                fil.write(out)
            else:
                path_split = file_path.split('/')
                path_all = ''
                for p in path_split[:-1]:
                    path_all += p + '/'
                    if not os.path.exists(path_all):
                        os.system('mkdir ' + path_all)
                fil = open(file_path,'w')
                fil.write(out)
        fil.close()
    ### Get information    
    if "(if normalized by length of Chain_1" in out:
        k_1 = out.index("(if normalized by length of Chain_1")
        tms_1 = out[k_1-8:k_1-1]
    else:
        return None
    if "(if normalized by length of Chain_2" in out:
        k_2 = out.index("(if normalized by length of Chain_2")
        tms_2 = out[k_2-8:k_2-1]
    else:
        return None
    out_split = out.split('\n')
    for line in out_split:
        if 'Length of Chain_1:' in line:
            length_1 = int(line.strip(' residues').split(' ')[-1])
        elif 'Length of Chain_2:' in line:
            length_2 = int(line.strip(' residues').split(' ')[-1])
        elif 'Aligned length' in line:
            length_align = int(line.split(',')[0].split(' ')[-1])
            RMSD = float(line.split(',')[1].split(' ')[-1])
    return tms_1,tms_2,length_1,length_2,length_align,RMSD

def Alignment(pdb,ref,index):
    command_1 = './TMalign ' + pdb + ' ' + ref + ' -o ' + index
    os.system(command_1)
    return 0
