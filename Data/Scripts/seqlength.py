import numpy as np

aminoletter=['a','c','d','e','f','g','h','i','k','l','m','n','p','q','r','s','t','v','w','y']

s0=np.loadtxt("../version02/seq_train", dtype='str')
s1=np.loadtxt("../version02/seq_test_trad", dtype='str')
s2=np.loadtxt("../version02/seq_test_with", dtype='str')

num=0
leng=[]
for i in s0:
	leng.append(len(i))
	for j in i:
		if(j not in aminoletter):
			num+=1
			print j

print "total:",num

num=0
for i in s1:
	leng.append(len(i))
	for j in i:
		if(j not in aminoletter):
			num+=1
			print j

print "total:",num

num=0
for i in s2:

	if(len(i)<500):
		leng.append(len(i))
	for j in i:
		if(j not in aminoletter):
			num+=1
			print i

print "total:",num



import matplotlib.pyplot as plt

plt.figure()
print leng
plt.hist(leng, bins=300, normed=True)
plt.xlim([0,500])
plt.savefig("hist_seq_length.png", format='png')
plt.show()