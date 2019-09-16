import numpy as np
from numpy import linalg
from represent import delete_fold
from sklearn.cluster import AgglomerativeClustering
from numpy import linalg

class eigen:
	def __init__(self):
		self.x=0
	def center(self, X):

		n1,n2 = len(X), len(X[0])
		
		H1= np.identity(n1) - np.ones((n1,n1))/float(n1)
		H2= np.identity(n2) - np.ones((n2,n2))/float(n2)

		return np.matmul(np.matmul(H1, X), H2) 

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

def get_score(X):

	if(len(np.array(X).shape) ==1 ):
		return [o.tm for o in X]
	else:
		return [[o.tm for o in X[i]] for i in range(len(X))]

def per50(w, threhold=0.5):
	wnorm= w/np.sum(w)
	
	for i in range(len(w)):
		if(np.sum(wnorm[0:i+1])>threhold):
			print ("%d	%d\n" %(i, len(w))),
			break

tm_old = np.loadtxt("../TM_matrix.fa")

bound=[]
cl=['a','b','c','d','e','f','g']


classs=[]
folds=[]
for lines in open("../represent_file"):
	classs.append(lines[5])

folds=np.loadtxt("folds_coordinate", dtype='str').T

'''
import nltk
import sklearn

print('The nltk version is {}.'.format(nltk.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
exit(0)
'''

class foldij:
	def __init__(self):
		self.crow = 'a'
		self.ccol = 'a'
		self.tm = 0.
		self.crow_name=''
		self.ccol_name=''


tm_mat = [[foldij() for j in range(len(tm_old))] for i in range(len(tm_old))]

for i in range(len(tm_old)):
	for j in range(len(tm_old)):
		tm_mat[i][j].crow = classs[i]
		tm_mat[i][j].ccol = classs[j]
		tm_mat[i][j].tm   =  tm_old[i][j]
		tm_mat[i][j].crow_name=folds[0][i]
		tm_mat[i][j].ccol_name=folds[0][j]

tm = delete_fold(tm_mat, 0.45, True)


tm_score=get_score(tm) 
print (len(tm_score), len(tm_score[0]))

total = eigen()
tm_score = total.center(tm_score)
w,v = total.fit(tm_score)
print ("all:", w[0], np.min(w), "len:", len(w))
per50(w, 0.25)


# kernel PCA-----------------------------------------

tm_new={}
tm_clas={}
basis_c={}
s=0
for c in ['a','b','c','d','e','f','g']:

	tm_new[c]=[]

	for i in range(len(tm)):

		if(tm[i][0].crow==c):
			tm_new[c].append([])
			for j in range(len(tm)):
				if(tm[i][j].ccol==c):
					tm_new[c][-1].append(tm[i][j])

	#if c=='d':
	#	tm_new = delete_fold(tm_new, 0.38)

	tm_clas[c]=tm_new[c]
	temp=get_score(tm_new[c])

	model=eigen()
	tem = model.center(temp)
	w,v = model.fit(tem)

	print ("class "+c+" top eigenvalue:", w[0]/len(w), w[1]/len(w), w[2]/len(w), "min: ", w[-1]/len(w))
	all7=[]
	for i in range(len(w)):
		temp=[]
		for j in range(3):                 #    class level choose first 3 eigenvalues
			temp.append((np.dot(tem[i], v[j])/np.sqrt(w[j])))
		all7.append(temp)

	print ("variance along top 3 axis: ", np.var(all7, axis=0))

	per50(w)

# store basis files:
	basis=[]
	for i in range(25):
		basis.append(v[i]/np.sqrt(w[i]))

	basis_c[c]=basis


print ("number of new folds:", s)


#--------------------------------------------7 representive folds:

t=0
s=[0. for i in range(7)]
index=[0 for i in range(7)]
l1=0
u1=0
for i in range(len(tm)):
	if(tm[i][0].crow == cl[t]):
		temp = np.sum([o.tm for o in tm[i]])

		if(temp > s[t]):
			s[t]=temp
			index[t]= i

	else:
		t+=1
		s[t]=temp
		index[t]= i



columns=np.array([index for i in range(7)])
rows=columns.T


rc = np.array(tm)[[rows, columns]]

tm_new = [[o.tm for o in rc[i]] for i in range(len(rc))]

#tm_new = tm_new[0:2].T[0:2]

model = eigen()
tm_new = model.center(tm_new)
w, v = model.fit(tm_new)

print (w[0]/7., w[1]/7., w[2]/7., w[3]/7. , w[4]/7., w[5]/7., w[6]/7.)


all7=[]
for i in range(7):
	temp=[]
	for j in range(6):                 #    class level choose first 3 eigenvalues
		temp.append((np.dot(tm_new[i], v[j])/np.sqrt(w[j])))
	#	temp.append(v[j][i]*np.sqrt(w[j]))
	

	print (w[i], np.var(v[i]))
	all7.append(temp)

print (np.var(all7, axis=0))
#print (v[0].dot(v[0]))
#----------------------------------------------------------------------------print results----------------------
np.savetxt("../data_v04/class7_tm", tm_new, fmt="%f ")
np.savetxt("../data_v04/class7_basis", all7, fmt="%f ")

folds=[]
with open("../data_v04/fold_basis","w") as f:
	for t in range(7):
		f.write("%c (%d %d)\n" %(cl[t], len(basis_c[cl[t]]), len(basis_c[cl[t]][0])))


		for j in range(len(basis_c[cl[t]][0])):
			f.write(tm_clas[cl[t]][j][0].crow_name+" ")
			folds.append(tm_clas[cl[t]][j][0].crow_name)


		f.write("\n")
		for i in range(len(basis_c[cl[t]])):
			for j in range(len(basis_c[cl[t]][0])):
				f.write("%f " %(basis_c[cl[t]][i][j]))

			f.write("\n")
		f.write("\n")



#represent file

covert={
	'a':0,
	'b':1,
	'c':2,
	'd':3,
	'e':4,
	'f':5,
	'g':6
}


with open("../data_v04/folds_coordinate","w") as f:
	for i in range(len(tm_mat)):
		vector=[]
		for j in range(len(tm_mat)):
			if(tm_mat[i][j].crow==tm_mat[i][j].ccol and  tm_mat[i][j].ccol_name in folds):
				vector.append(tm_mat[i][j].tm)


		index = covert[tm_mat[i][0].crow]
		cc =   tm_mat[i][0].crow


		coor = all7[index].copy()
		for j in range(len(basis_c[cc])):
			coor.append(np.dot(vector, basis_c[cc][j]))

		#print (len(vector), len(basis_c[cc][4]))
		

		f.write(tm_mat[i][0].crow_name+" ")
		for j in range(len(coor)):
			f.write("%11.6f" %(coor[j]))

		f.write("\n")


exit(0)



#--------------------hirachycal clustering----------------------------------------------
'''
hac = AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage='complete')

hac.fit(dis_mat['a'])

n_samples = len(dis_mat['a'])


print (hac.children_)
np.savetxt("tree_hac", hac.children_, fmt='%d')


child = hac.children_
fa=[-1 for i in range(n_samples*2 -1)]

for i in range(n_samples-1):
	fa[child[i][0]] = i+ n_samples
	fa[child[i][1]] = i+ n_samples

	if(child[i][0]<n_samples and child[i][1]<n_samples):
		print tm[child[i][0]][child[i][1]]

max_deep=0
index=0

print fa
for i in range(n_samples):
	l=0
	leave=i
	while leave!=-1 :
		leave=fa[leave]
		l+=1

	if(l>max_deep):
		max_deep=l
		index = i
'''
#--------------------hirachycal clustering----------------------------------------------



# DBSCAN--------------------------------------------
from sklearn.cluster import DBSCAN

for c in cl:
	tm_new =  np.array( delete_fold(-dis_mat[c], 0.5) )

	tm_new = np.ones(tm_new.shape) - tm_new


	for ep in ['0.7']:
		model = DBSCAN(eps = float(ep), min_samples=1, metric='precomputed')
		model.fit(tm_new)
		print (np.unique(model.labels_))









