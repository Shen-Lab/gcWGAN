import numpy as np
import os
from numpy import linalg

def TM_score(pdb_1,pdb_2):
    command_1 = './TMalign ' + pdb_1 + ' ' + pdb_2 + ' -a'
    output_1 = os.popen(command_1)
    out_1 = output_1.read()
    if "(if normalized by length of Chain_1)" in out_1:
        k_1 = out_1.index("(if normalized by length of Chain_1)")
        tms_1 = out_1[k_1-8:k_1-1]
    else:
        return None
    
    command_2 = './TMalign ' + pdb_2 + ' ' + pdb_1 + ' -a'
    output_2 = os.popen(command_2)
    out_2 = output_2.read()
    if "(if normalized by length of Chain_1)" in out_2:
        k_2 = out_2.index("(if normalized by length of Chain_1)")
        tms_2 = out_2[k_2-8:k_2-1]
    else:
        return None

    return (float(tms_1) + float(tms_2))/2

def TM_1200(pdb,path):
    l = len(os.listdir(path))
    #print(l)
    result = []
    for i in range(1,l+1):
        tms = TM_score(pdb,path + str(i) + '.pdb')
        #print(tms)
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

    tm = np.loadtxt("../Data/Datasets/Final_Data/TM_matrix.fa")
    tm = tm + np.identity(len(tm))*5.6
    tm_test = TM_1200(pdb, '../Data/Datasets/Final_Data/pdbs/')

    model=eigen()
    tm_test_new = model.center(np.array([tm_test]), tm)
    fold_basis=np.loadtxt("../Data/Datasets/Final_Data/fold_basis")

    new_coor=[]

    for i in range(1):
        coor_temp=[]
        for j in range(20):
                coor_temp.append(np.dot(tm_test_new[i],fold_basis[j]))

        new_coor.append(coor_temp)
    return np.array(new_coor)

def TM_score_ref(ref,pdb):
    command_1 = './TMalign ' + ref + ' ' + pdb + ' -a'
    output_1 = os.popen(command_1)
    out_1 = output_1.read()
    if "(if normalized by length of Chain_1)" in out_1:
        k_1 = out_1.index("(if normalized by length of Chain_1)")
        tms_1 = out_1[k_1-8:k_1-1]
    else:
        return None

    return float(tms_1)

def TM_1200_ref(path,pdb):
    l = len(os.listdir(path))
    #print(l)
    result = []
    for i in range(1,l+1):
        tms = TM_score_ref(path + str(i) + '.pdb',pdb)
        #print(tms)
        result.append(tms)
    return result

