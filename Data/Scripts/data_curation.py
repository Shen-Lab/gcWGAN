import numpy as np

class seq:
	c=''


folds={}

pearsefile=open("dir.des.scope.2.07-stable.txt")
astralfile=open("astral-scopedom-seqres-gd-all-2.07-stable.fa")

for lines in pearsefile:
	line=lines.strip('\n').split()
	if(line[1]=="cf"):
		folds[line[2]]=line[4:]



seqs={}
for lines in astralfile:
	line=lines.strip('\n')
	
	seq=''
	if(line[0]=='>'):
		bs=1
	else:
		seq+=line