import numpy as np
from numpy import linalg
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
threhold=0.4
#delete abnormal folds
class eigen:
	def __init__(self):
		self.x=0
	def center(self, X, tm):

		m,n = len(X), len(X[0])
		
		H1= np.ones((n,n))/float(n)
		H2= np.ones((m,n))/float(n)


		#y = X - np.matmul(X, H1) - np.matmul(H2,tm) + np.matmul(np.matmul(H2,tm), H1)
		#print (np.mean(np.diagonal(X)), np.std(np.diagonal(X)), np.mean(np.diagonal(y)), np.std(np.diagonal(y)) )
		#exit(0)

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
def delete_fold(tm, threhold=0.5, tiao=False):
	
	ad={}

	n=[0 for i in range(len(tm))]
	sum=0

	for i in range(len(tm)):
		for j in range(i+1, len(tm)):

			if(tm[i][j].tm>threhold  or (tiao==True and tm[i][j].crow=='d' and tm[i][j].ccol=='d' and tm[i][j].tm>0.38)):
				if(i not in ad):
					ad[i]=[j]
				else:
					if(j not in ad[i]):
						ad[i].append(j)


				if(j not in ad):
					ad[j]=[i]
				else:
					if(i not in ad[j]):
						ad[j].append(i)

				n[i]+=1
				n[j]+=1
				sum+=1

	judge=[0 for i in range(len(tm))]

#	print "how many edges", sum, sum/float(len(tm)**2)

	while(sum!=0):
		index=n.index(np.max(n))

		judge[index]=1


		for edge in ad[index]:
			n[edge]-=1

		sum-=n[index]
		n[index]=0

	#print np.sum(judge)

	tm_new=[]

	for i in range(len(tm)):
		if(judge[i]==1):
			continue
		row=[]
		for j in range(len(tm)):
			if(judge[j]==1):
				continue 

			row.append(tm[i][j])

		tm_new.append(row)

	#print "new_tm_matrix_length:", len(tm_new)
	return tm_new
#return 0

def plot_hist(x1,x2,x3, name1,name2, name3):
	plt.hist(x1,bins=100, normed='True',label=name1)
	plt.hist(x2,bins=100, normed='True',label=name2)
	plt.hist(x3,bins=100, normed='True',label=name3)
	plt.legend(loc='upper right')
	plt.savefig("tm_dis",  format='png')
	plt.show()

def plot_relation(x1,x2,x3, name1,name2, name3):

	plt.figure()
	plt.scatter(x1,x2, label=["x="+name1, "y="+name2])
	plt.legend(loc='upper right')
	plt.savefig("tm1tm2",format='png')
	plt.show()
	plt.close()

	plt.figure()
	plt.scatter(x1,x3, label=["x="+name1, "y="+name3])
	plt.legend(loc='upper right')
	plt.savefig("tm1tm3",format='png')
	plt.show()
	plt.close()

	plt.figure()
	plt.scatter(x2,x3, label=["x="+name2, "y="+name3])
	plt.legend(loc='upper right')
	plt.show()
	plt.close()





def re():

	#tm = np.load("TM_score_matrix.npy")
	#tm = np.loadtxt("../Datasets/Fold_representation/TM_score/Gram_Matrix/TM_matrix/TM_matrix.fa")
	tm = np.load("TM_matrix_2_pdbs-noTER_symmetric.npy")
	print (tm.shape)

	#tm = tm + np.identity(len(tm))*2.50
	

	model=eigen()
	
	tm_flat = np.reshape(tm, (-1,))
	print ("original:", np.mean(tm_flat), np.std(tm_flat))
	#plot_hist(tm_flat, "original")
	
	from nearest_correlation import nearcorr

	print ("Frobenius norm:", np.linalg.norm( tm-nearcorr(tm, max_iterations=2000, tol=[1E-5]), 'fro'))

	tm = nearcorr(tm, max_iterations=2000, tol=[1E-5])


	#tm = tm + np.identity(len(tm))*0.001
	#print ("w", np.mean(w), np.std(w), np.min(w))
		
	w,v = model.fit(tm)
	print ("w", np.mean(w), np.std(w), np.min(w))
	tm_flat1 = np.reshape(tm, (-1,))
	print ("before_centralized", np.mean(tm_flat1), np.std(tm_flat1))
	#plot_hist(tm_flat1, "before_centralized")
	print (sum(w[0:20])/sum(w))
	
	tm = model.center(tm,tm)
	#tm = tm + np.identity(len(tm))*0.000001
	tm = tm + np.identity(len(tm))*0.001
	tm_flat2 = np.reshape(tm, (-1,))
	print ("after centerlized", np.mean(tm_flat2), np.std(tm_flat2))
	#plot_hist(tm_flat, tm_flat1, tm_flat2, "original", "before_centralized", "after_centralized")
	#plot_relation(tm_flat, tm_flat1, tm_flat2, "original", "before_centralized", "after_centralized")
	w,v = model.fit(tm)
	print ("w", np.mean(w), np.std(w), np.min(w))
	print (sum(w[0:20])/sum(w))
	'''	
	sum_ne=0
	for i in range(len(tm)):
		if(w[i]<0):
			w[i]=0.001
			sum_ne+=1
	#print (sum_ne, np.linalg.norm(v[0]), np.linalg.norm(v[33]))
	
	tm_flat = np.reshape(tm, (-1,))
	print ("previous", np.mean(tm_flat), np.std(tm_flat))

	tm = np.zeros((len(tm), len(tm)))

	for i in range(len(tm)):
		ttt=np.reshape(v[i], (1, len(tm)))
		y = np.multiply(ttt.T, ttt*w[i])
		tm+=y
	# geting the new tm matrix after truncation
	tm = model.center(tm,tm)
		
	w,v = model.fit(tm)


	tm_flat = np.reshape(tm, (-1,))
	print ("now", np.mean(tm_flat), np.std(tm_flat))
	'''
	np.save("../Results/eigenvalues", np.array(w))
	np.save("../Results/eigenvectors", np.array(v))
	#w=np.load("Results/eigenvalues.npy")
	#v=np.load("Results/eigenvectors.npy")
	print (sum(w[0:20])/sum(w))
	print ("min:", min(w))


	
	import matplotlib.pyplot as plt
	plt.figure()	

	z1=np.linspace(-500, 20)
	z=np.linspace(0,1.1)
	x=np.arange(0, 1232, 1)
	for i in range(len(x)):
		plt.scatter(x[i], np.sum(w[0:(i+1)])/np.sum(w), color='blue' , s=4)
	plt.plot(np.zeros(z.shape)+20, z , linestyle='--', color='black')
	plt.plot(z1, np.zeros(z1.shape)+np.sum(w[0:20])/ np.sum(w) , linestyle='--', color='black')
	
	plt.xlabel("Index", fontsize=12)
	plt.ylabel("Cumulative explained Variance", fontsize=12)
	plt.ylim([0,1.1])
	plt.xlim([-100,1400])
	plt.xticks([0,  200, 400, 600, 800, 1000, 1200],(0, 200, 400, 600, 800, 1000, 1200))
	plt.yticks([0.0, 0.20, 0.4, 0.6, 0.8, 1], (0.0, 0.20,0.4,0.6,0.8,1))
	plt.savefig("../Results/explained_variance.eps", format='eps')
	plt.savefig("../Results/explained_variance.png", format='png')
	plt.show()
	plt.close()
	


	coor=[]
	dis=np.zeros((len(tm), len(tm)))
	
	for i in range(len(tm)):

		temp=[]
		temp1=[]
		#temp.append(fold[i])

		for j in range(len(tm)):
			temp.append(str("%7.4f" %(np.dot(tm[i], v[j])/np.sqrt(w[j]))))
			temp1.append(np.dot(tm[i], v[j])/np.sqrt(w[j]))

		#top20.append(temp)
		coor.append(temp1)

	folds_name = np.loadtxt("folds_name", dtype='str')
	with open("../Results/folds_coordinate", "w") as f:
		for i in range(len(tm)):
			f.write(folds_name[i])
			f.write(" ")
			for element in coor[i][0:20]:
				f.write("%6.3f " %(element))
			f.write("\n")


	# save coordinates ---------------start


	basis=[]

	for i in range(20):
		basis.append(v[i]/np.sqrt(w[i]))
		
	np.savetxt("../Results/folds_basis", basis, fmt='%s')


	# save coordinates------------------end




	for i in range(20):
		print (np.var(coor, axis=0)[i], w[i]/1232.)

	
	
	#coor=np.loadtxt("Results/folds_coordinate")
	from sklearn.cluster import KMeans
	def fold_center_var(ncluster):

		kmeans = KMeans(n_clusters=ncluster, random_state=0).fit(coor)
		coor100 = kmeans.cluster_centers_       # coor100 shape (ncluster, 1232)

		pca = PCA()
		pca.fit(coor100)

		 # coor100 is the cluster heads coordinates 

		# we only need to consier two space: 1232d and n_cluster-dimensinal space
		# project the original first 20 basis into these n_cluster-spanned subspace, which are [1,0...], [0,1...].. in 1232d space
		pbasis = pca.components_   #pbasis (n_cluster, 1232)
		

		coor20pbasis = []  # shape(20, n_cluster)
		for i in range(20):
			obasis = np.zeros((1232,))
			obasis[i]=1.

			cw = []
			for j in range(len(pbasis)):
				cw.append(np.dot(obasis, pbasis[j])/np.linalg.norm(pbasis[j]))
			
			coor20pbasis.append(cw)

			

		
		# calculate variance of n_cluster cluster heads along these 20 basis in n_clusterd

		#1. first calculate the n_cluster cluster heads coordinate in n_cluster d

		coorncluster = []   # shape(n_cluster, n_cluster)
		for i in range(ncluster):
			cw = []
			for j in range(ncluster):
				cw.append( np.dot(coor100[i], pbasis[j])/np.linalg.norm(pbasis[j]) )
			coorncluster.append(cw)


		# then project those points along these 20 dimensional vectors.
		nominator = 0.
		for i in range(20):
			cw = []
			for j in range(ncluster):
				cw.append( np.dot(coor20pbasis[i], coorncluster[j])/np.linalg.norm(coor20pbasis[i]) )

			nominator+=np.var(cw)		
		denominator = sum(pca.explained_variance_)
		print (nominator/denominator)
		return nominator/denominator
		




	x = [20, 30, 40, 60, 80, 100, 120, 140, 160, 180, 200]
	y = []
	for xx in x:
		y.append(fold_center_var(xx))
	import matplotlib.pyplot as plt
	plt.plot(x, y)
	plt.yticks(fontsize=14)
	plt.ylim([0,1.1])
	plt.xlim([0, 200])
	plt.xticks(x, x)
	plt.xlabel("#clusters",  fontsize=14)
	plt.ylabel("Cumulative explained Variance", fontsize=12)
	plt.savefig("../Results/explained_variance_after_clustering.eps", format='eps')
	plt.savefig("../Results/explained_variance_after_clustering.png", format='png')
	plt.show()

	exit(0)


	'''
	exit(0)


	z1=np.linspace(-100,20)

	z=np.linspace(0,1)
	plt.plot(np.zeros(z.shape)+20, z , linestyle='--', color='black')

	plt.plot(z1, np.zeros(z1.shape)+np.sum(w[0:20])/np.sum(w), linestyle='--', color='grey')
#	plt.text(22, np.sum(w[0:20])/np.sum(w), "[%.1f, 20]" %())
	for i in range(len(x)):
		plt.scatter(x[i], np.sum(w[0:(i+1)])/np.sum(w), color='blue' , s=4)

	print np.sum(w[0:20])/np.sum(w)
	#plt.xticks(x, (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20))
	plt.xlabel("Index", fontsize=12)
	plt.ylabel("Cumulative explained Variance", fontsize=12)
	plt.ylim([0,1])
	plt.xlim([-100,1400])
	plt.xticks([0,  200, 400, 600, 800, 1000, 1200],(0, 200, 400, 600, 800, 1000, 1200))
	plt.yticks([0.0, 0.20, 0.4, 0.6, 0.8, 1], (0.0, 0.20,0.4,0.6,0.8,1))
	plt.savefig("curve.eps", format='eps')
	plt.show()
	plt.close()
	exit(0)
	'''
	classs=[]
	fold=[]
	top20=[]
	for lines in open("../represent_file"):
		classs.append(lines[5])

		for b in range(8, 12):
			if(lines[b]=='.'):
				fold.append(lines[5:b])
				break

#	print fold, len(fold)
	
	coor=[]
	dis=np.zeros((len(tm), len(tm)))

	for i in range(len(tm)):

		temp=[]
		temp1=[]
		temp.append(fold[i])

		for j in range(500):
			temp.append(str("%7.4f" %(np.dot(tm[i], v[j])/np.sqrt(w[j]))))
			temp1.append(np.dot(tm[i], v[j])/np.sqrt(w[j]))

		top20.append(temp)
		coor.append(temp1)

	from sklearn.cluster import KMeans
	kmeans = KMeans(n_clusters=100, random_state=0).fit(coor)
	var = np.var(np.array(kmeans.cluster_centers_), axis=0)
	print np.sum(var[0:20])/np.sum(var[0:100])
#	for i in range(1232):
#		print ("%.2f %.3f %.3f" % (w[i], w[i]/len(w), np.var(np.array(coor).T[i]) ))

	exit(0)

	np.savetxt("folds_coordinate", top20, fmt='%s')

	#basis
	basis=[]

	for i in range(20):
		basis.append(v[i]/np.sqrt(w[i]))
		
	np.savetxt("folds_basis", basis, fmt='%s')	

	for i in range(len(w)):
		print w[i]

	minm=1.0
	'''
	for i in range(len(tm)):
		for j in range(len(tm)):
			dis[i][j]=np.linalg.norm(np.subtract(coor[i],coor[j]))
			if(i!=j and dis[i][j]<minm):
				minm=dis[i][j]
			print ("%.4f " %(dis[i][j])),

		print ('\n')
'''
	

	for i in range(20):
		print ("%.2f %.3f %.3f" % (w[i], w[i]/len(w), np.var(np.array(coor).T[i]) ))









	exit(0)
#----------------1/11/2019----------------------------

	for threhold in np.linspace(0.3, 1, 8):


		tm_new=delete_fold(tm)

		#print len(tm_new), len(tm_new[0])

		w, v = linalg.eig(tm_new)
		#print np.min(w)

		if(np.min(w)<0):
			tm_new = np.add(tm_new, -np.min(w)*np.identity(len(tm_new)))

		w, v = linalg.eig(tm_new)
		v=np.transpose(v)
		#print np.min(w)

		for i in range(len(w)):
			for j in range(i+1, len(w)):
				if(w[i]<w[j]):
					temp=w[i]
					w[i]=w[j]
					w[j]=temp
					temp=v[i]
					v[i]=v[j]
					v[j]=temp

		wnorm= w/np.sum(w)

		for i in range(len(tm_new)):
			if(np.sum(wnorm[0:i+1])>0.5):
				print ("%.3f	%.0f	%.0f\n" %(threhold, i, len(tm_new))),
				break




	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import axes3d




	plt.figure()

	x=np.arange(0, 20, 1)

	for i in range(50):
		print ("%d	 %.4f	%.4f" %(i+1, wnorm[i], np.sum(wnorm[0:i+1])))


	exit(0)

	for i in range(len(x)):
		plt.bar(x[i], w[i], color='blue', width=0.7)

	#plt.savefig("top20eigv.png", format='png')
	plt.close()

	x1=[]
	x2=[]
	x3=[]

	top20=[]
	for i in range(len(tm)):
		x1.append(np.dot(tm[i], v[0])/np.sqrt(w[0]))
		x2.append(np.dot(tm[i], v[1])/np.sqrt(w[1]))
		x3.append(np.dot(tm[i], v[2])/np.sqrt(w[2]))



		#print x1[-1],x2[-1],x3[-1], v[0][i]*np.sqrt(w[0]), v[1][i]*np.sqrt(w[1]), v[2][i]*np.sqrt(w[2])


	classs=[]
	fold=[]
	for lines in open("../represent_file"):
		classs.append(lines[5])

		for b in range(8, 12):
			if(lines[b]=='.'):
				fold.append(lines[5:b])
				break

#	print fold, len(fold)

	for i in range(len(tm)):

		temp=[]

		#temp.append(fold[i])

		for j in range(20):
			temp.append(str("%7.4f" %(np.dot(tm[i], v[j])/np.sqrt(w[j]))))

		top20.append(temp)

	#np.savetxt("folds_coordinate", top20, fmt='%s')

	#basis
	basis=[]

	for i in range(20):
		basis.append(v[i]/np.sqrt(w[i]))
	#np.savetxt("fold_basis", basis)



	c=[]

	colormap='rainbow'

	class1={
		'a':'red',
		'b':'pink',
		'c':'orange',
		'd':'yellow',
		'e':'green',
		'f':'blue',
		'g':'purple'
	}



	fig=plt.figure()
	ax = fig.gca(projection='3d')

	label={
		'a':r'$\alpha$',
		'b':r'$\beta$',
		'c':r'$\alpha/\beta$',
		'd':r'$\alpha+\beta$',
		'e':r'$\alpha and \beta$',
		'f':'Membrane and cell surface',
		'g':'Small proteins'
	}
	judge={}
	for i in range(len(x1)):

		if classs[i] in judge:
			ax.scatter(x1[i],x2[i],x3[i], alpha=0.8, c=class1[classs[i]])
		else:
			ax.scatter(x1[i],x2[i],x3[i], alpha=0.8, c=class1[classs[i]], label=label[classs[i]])
			judge[classs[i]]=1

	ax.set_xlabel("PC1")
	ax.set_ylabel("PC2")
	ax.set_zlabel("PC3")
	plt.legend(loc="upper left")

	#plt.close()
	fig=plt.figure()
	ax = fig.gca(projection='3d')


	withold=np.loadtxt("withhold_folds", dtype='str')


	j=0
	for i in range(len(x1)):

		if fold[i] in withold:
			j+=1
#			print j
			ax.scatter(x1[i],x2[i],x3[i], alpha=0.8, c='blue')
		else:
			ax.scatter(x1[i],x2[i],x3[i], alpha=0.8, c='green')
			

	ax.set_xlabel("PC1")
	ax.set_ylabel("PC2")
	ax.set_zlabel("PC3")
	plt.legend(loc="upper left")


	#plt.show()

	plt.close()

	# t-SNE
	from sklearn.manifold import TSNE

	t_embed=TSNE(n_components=3).fit_transform(top20)

#	print t_embed.shape

	judge={}
	for i in range(len(x1)):

		if classs[i] in judge:
			ax.scatter(t_embed[i][0],t_embed[i][1],t_embed[i][2], alpha=0.8, c=class1[classs[i]])
		else:
			ax.scatter(t_embed[i][0],t_embed[i][1],t_embed[i][2], alpha=0.8, c=class1[classs[i]], label=label[classs[i]])
			judge[classs[i]]=1

	ax.set_xlabel("PC1")
	ax.set_ylabel("PC2")
	ax.set_zlabel("PC3")
	plt.legend(loc="upper left")

	plt.show()


re()
