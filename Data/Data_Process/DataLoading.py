import numpy as np

def Interval_dic(path):
    """
    Load Interval file and return a dictionary.
    Keys are folds name.
    Values are lists, where first element is the fold number and others are sequences.
    """
    fil = open(path)
    lines = fil.readlines()
    dic = {}
    for i in lines:
        line = i.strip('\n')
        if line[1] == '.' :
            line = line.split('\t')
            fold = line[0]
            dic[fold] = [int(line[1])]
        else:
            dic[fold].append(line)
    fil.close()
    return dic

def Train_dic(path_1,path_2):
    folds = open(path_1,'r')
    seqs = open(path_2,'r')
    lines_f = folds.readlines()
    lines_s = seqs.readlines()
    if len(lines_f) != len(lines_s):
        print 'Input unrelated files!'
        return None
    else:
        f_s_dic = {}
        for i in xrange(len(lines_f)):
            f = lines_f[i].strip('\n')
            s = lines_s[i].strip('\n')
            if f in f_s_dic.keys():
                f_s_dic[f].append(s)
            else:
                f_s_dic[f] = [s]
    return f_s_dic

def file_dic(seq_path,fold_path):
    """
    Load test file and return a dictionary.
    Keys are folds name.
    Values are lists, where first element is the fold number and others are sequences.
    """
    s_file = open(seq_path)
    f_file = open(fold_path)
    s_lines = s_file.readlines()
    f_lines = f_file.readlines()
    l_s = len(s_lines)
    l_f = len(f_lines)
    if l_s != l_f:
        print "Input wrong file"
        return 0
    dic = {}
    for i in xrange(l_s):
        fold = f_lines[i].strip('\n')
        seq = s_lines[i].strip('\n')
        if fold in dic.keys():
            dic[fold].append(seq)
        else:
            dic[fold] = [seq]
    s_file.close()
    f_file.close()
    return dic

def file_list(path):
    f = open(path,'r')
    lines = f.readlines()
    result = []
    for i in lines:
       line = i.strip('\n')
       result.append(line)
    return result

def representative_dic(path,dic):
    d_c = {}
    fil = open(path,'r')
    lines = fil.readlines()
    l = len(lines)
    i = 0
    while(i < l):
        if  lines[i][1] == '.':
            fold = lines[i].strip('\n')
            index = lines[i+1].strip('\n').split(' ')[:-1]
            r_seq = [dic[fold][int(j)] for j in index]
            if not (fold[0] in d_c):
                d_c[fold[0]] = {fold:r_seq}
            else:
                d_c[fold[0]][fold] = r_seq
            i += 3
        else:
            i += 1
    fil.close()
    if len(d_c.keys()) == 7 and l == len(dic.keys())*3:
        return d_c
    else:
        print 'Error! Wrong folds number!'
        return 0

def Accuracy_index(path = 'DeepSF_model_weight/DeepSF_fold.txt'):
    fil = open(path,'r')
    lines = fil.readlines()
    d_c = {}
    for line in lines[1:]:
        l = line.strip('\n').split('\t')
        d_c[int(l[1])] = l[0]
    return d_c

def LookUpTable(path = 'Pretrain/folds_diatance_2'):
    dic = {}
    fil = open(path,'r')
    lines = fil.readlines()
    fil.close()
    f_list = lines[0].strip('\n').split('\t')
    f_list = f_list[2:]
    for line in lines[1:]:
        line = line.strip('\n').split('\t')
        dic[line[0]] = {}
        for j in range(len(f_list)):
            dic[line[0]][f_list[j]] = float(line[j+1])
    return dic

def one_to_one_dic(path):
    """
    The input file has two columns.
    Items in the first column will be the keys and
    those in the other column will be the values.
    """
    fil = open(path,'r')
    lines = fil.readlines()
    fil.close()
    result = {}
    for line in lines:
        line = line.strip('\n').split('\t')
        result[line[0]] = line[1]
    return result

def columns_to_lists(path):
    """
    Input a file with n columns.
    Return n lists of the columns.
    """
    fil = open(path,'r')
    lines = fil.readlines()
    fil.close()
    n = len(lines[0].strip('\n').split('\t'))
    result = []
    for i in range(n):
        result.append([])
    for line in lines:
        line = line.strip('\n').split('\t')
        for i in range(n):
            result[i].append(line[i])
    return result
   

def read_repredata(line):
    '''
    Extract the pdb, fold, chain and residue infomation in a line.
    '''
    pdb = line[0:4]
    fold = line.split(' ')[1]
    fold = fold.split('.')[0] + '.' + fold.split('.')[1]

    result = []
    line = line.split('(')
    for j in line[1:]:
        info_dict = {'pdb_id':pdb,'fold':fold}
        info = j.split(')')[0].split(' ')
        info_dict['chain'] = info[1]
        if len(info) >= 5:
            resi = info[-1]
            if resi[0] == '-':
                resi = [r for r in resi.split('-') if r != '']
                resi[0] = '-' + resi[0]
                info_dict['resi'] = resi
            else:
                info_dict['resi'] = [r for r in resi.split('-')]
        else:
            info_dict['resi'] = None
        result.append(info_dict)
    return result 
