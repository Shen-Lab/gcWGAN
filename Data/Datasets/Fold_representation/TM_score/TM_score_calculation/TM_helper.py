import numpy as np
import os

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
    print(l)
    result = []
    for i in range(1,l+1):
        tms = TM_score(pdb,path + str(i) + '.pdb')
        #print(tms)
        result.append(tms)
    return result
