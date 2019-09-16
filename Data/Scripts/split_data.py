import numpy as np

aminoletter=['a','c','d','e','f','g','h','i','k','l','m','n','p','q','r','s','t','v','w','y']
nn_aa_possbility={
				'b':['d','n'],
				'z':['q','e'],
				'x': ['a','c','d','e','f','g','h','i','k','l','m','n','p','q','r','s','t','v','w','y']
}

all_folds_to_seq={}
delete_folds_to_seq=[]

def sanity_check(seq):

	b=2
	for letter in seq:
		if(letter not in aminoletter):

			if(letter not in nn_aa_possbility):
				return 0  # containing 'X', must delete seq immediately 
			else:
				b=1  #  containing 'b','z', or 'x',  delete in version01, replaced by random sampling in version02 in no 'X' further detected

	return b              #  b=2: containing only 20 AA, keep it;   b=1: containing 'b','z', or 'x' no 'X'

def aa_replace_random(seq):

	new=''
	for l in seq:
		if l in aminoletter:
			new+=l
		else:
			new+=np.random.choice(nn_aa_possbility[l], 1, replace=True)[0]

	return new


def fold(path):
	

	f=open(path)
	while(1):

		lines = f.readline()
		if(lines==''):
			break

		line1 = lines.strip('\n')

		if(line1[1]=='.'):
			line = line1.split()

			all_folds_to_seq[line[0]]=[]
			for i in range(int(line[1])):

				lines=f.readline()
				lines=lines.strip('\n')

#version0.1
#-------------------------------------------------If the sanity_check not pass(non_natrual amino-acid, it will delete)
#version0.2
#-------------------------------------------------If the sanity_check not pass(non_natrual amino-acid, it will random sample)
				
				judge=sanity_check(lines)

				
				
				if(judge == 0):
					delete_folds_to_seq.append([line[0], lines])
					continue

				if(judge == 1):
					#delete_folds_to_seq.append([line[0], lines])
					all_folds_to_seq[line[0]].append(aa_replace_random(lines))  #------------version 2
					continue    


				if(judge == 2):
					all_folds_to_seq[line[0]].append(lines)
					continue

								#-----------------version 1

				

			if(all_folds_to_seq[line[0]]==[]):
				del all_folds_to_seq[line[0]]              #-------------------------------delete empty fold
				print  "deleting folds",line[0]

	f.close()

'''
diff="../old_classfied_data/Difficult.fa"
med ="../old_classfied_data/Median.fa"
ez  ="../old_classfied_data/Easy.fa"

fold(ez)
fold(med)
fold(diff)
'''

fold("../version03/subset2/Interval_2.fa")

print len(all_folds_to_seq)

sq={}
for i in all_folds_to_seq:
	for seq in all_folds_to_seq[i]:
		
		if(seq in sq):
			print i

		sq[seq]=i

np.savetxt("delete_seqs", delete_folds_to_seq, fmt='%s')

#------------------------end data prepocessing-------------------
#----------------------------------------------------------------



#--------------start to split data using stratified sampling-----
#----------------------------------------------------------------
#----------------------------------------------------------------
#----------------------------------------------------------------


#stratified data
import copy
class1={
		'a':{},
		'b':{},
		'c':{},
		'd':{},
		'e':{},
		'f':{},
		'g':{}
}
class2= copy.deepcopy(class1)
class3= copy.deepcopy(class1)


seq_strat={
'ez': class1,
'med': class2,
'diff': class3
}

for i in all_folds_to_seq:

	if(len(all_folds_to_seq[i])>50 ):

		seq_strat['ez'][i[0]][i]=all_folds_to_seq[i]
		

	else:
		if(len(all_folds_to_seq[i])>5):

			seq_strat['med'][i[0]][i]=all_folds_to_seq[i]
		else:
			seq_strat['diff'][i[0]][i]=all_folds_to_seq[i]


#withhold=np.loadtxt("withhold_folds", dtype='str')


# ---------------------------------------------------statisitcs of seq_strat
s4=0
for difficult in seq_strat:

	s3=0
	for classs in seq_strat[difficult]:

		s2=0
		for folds in seq_strat[difficult][classs]:
			s2+=len(seq_strat[difficult][classs][folds])

		s3+=s2

		print difficult, classs, s2

	s4+=s3

	print s3

print 'end difficult'


withhold=[]
seq_test_with=[]
fold_test_with=[]



for difficult in seq_strat:
	for classs in seq_strat[difficult]:
		
		list1=seq_strat[difficult][classs].keys()
#-------------------------sample withhold testing set
#------------------------------------------------------
		try:
			withhold = np.concatenate((withhold, np.random.choice(list1, int(0.1*len(list1))+1, replace=False)),axis=0)
		except:
			print "empty: class", difficult, classs


		#print difficult, classs, int(0.1*len(list1))+1

for fold in withhold:
	for seq in all_folds_to_seq[fold]:

		fold_test_with.append(fold)
		seq_test_with.append(seq)


np.savetxt("withhold_folds", withhold,fmt='%s')
np.savetxt("seq_test_with", seq_test_with,fmt='%s')
np.savetxt("fold_test_with", fold_test_with,fmt='%s')

#seq_test_trad=np.loadtxt("seq_test_trad", dtype='str')



#-------------------------sample traditional testing set and the rest of seqs are in training set
#------------------------------------------------------
seq_test_trad=[]
fold_test_trad=[]

seq_train=[]
fold_train=[]

for difficult in seq_strat:

	s3=0
	for classs in seq_strat[difficult]:

		s2=0
		for folds in seq_strat[difficult][classs]:

			if(folds in withhold):
				continue

			list1=seq_strat[difficult][classs][folds]
			
			rd_seq= np.random.choice(list1, int(0.2*len(list1))+1, replace=False)

			seq_test_trad = np.concatenate((seq_test_trad, rd_seq),axis=0)
			fold_test_trad = np.concatenate((fold_test_trad, [folds for i in range(int(0.2*len(list1))+1)]), axis=0)

	#-------------------------------------put the rest into training set

			for seq in rd_seq:
				list1.remove(seq)

			seq_train = np.concatenate((seq_train, list1), axis=0)
			fold_train = np.concatenate((fold_train, [folds for i in range(len(list1))]), axis=0) 
	#---------------------------------------add training set

			s2+=int(0.2*len(list1))+1
		
		print difficult, classs, s2

		s3+=s2

	print difficult, s3


np.savetxt("seq_test_trad", seq_test_trad, fmt='%s')
np.savetxt("fold_test_trad", fold_test_trad, fmt='%s')

np.savetxt("seq_train", seq_train, fmt='%s')
np.savetxt("fold_train", fold_train, fmt='%s')





