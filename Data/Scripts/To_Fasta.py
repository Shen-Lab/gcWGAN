import sys

def seq_fasta(seq):
    seq = seq.upper()
    result = [">"]
    while(len(seq)>80):
	result.append(seq[0:80])
	seq = seq[80:]
    result.append(seq)
    return result

path = sys.argv[1]
path_2 = 'FASTA_' + path
path_3 = 'Test_FASTA_' + path
f = open(path,"r")
f_2 = open(path_2,"w")
f_3 = open(path_3,"w")
lines = f.readlines()

for s in lines:
    seq = s.strip('\n')
    fasta = seq_fasta(seq)
    for i in fasta:
        f_2.write(i+'\n')

for j in range(3):
    seq = lines[j].strip('\n')
    fasta = seq_fasta(seq)
    for i in fasta:
        f_3.write(i+'\n')
        
f.close()
f_2.close()
f_3.close()