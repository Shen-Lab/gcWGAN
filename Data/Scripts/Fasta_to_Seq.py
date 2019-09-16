import sys

path = sys.argv[1]

file_fasta = open(path,'r')
name = path.strip('.--.fatsa')[-10:]
file_seq = open('seq_50_' + name,'w')
#file_fold = open('pseudo_fold_' + name,'w')

lines = file_fasta.readlines()
l = len(lines)
charmap = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']

def validation(seq,charmap):
    result = True
    for e in seq:
        if not (e in charmap):
            result = False
    return result

i = 0
while(i < l):
    if lines[i][0] == '>':
        if (i != 0) and validation(seq,charmap):
            file_seq.write(seq.lower() + '\n')
            #file_fold.write('a.1\n')
        i += 1
        seq = ''
    else:
        seq += lines[i].strip('\n')
        i += 1
        if i == l:
            if validation(seq,charmap):
                file_seq.write(seq.lower() + '\n')
            #file_fold.write('a.1\n')

file_fasta.close()
file_seq.close()
#file_fold.close()
