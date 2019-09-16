import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np

fold_file = open('folds_coordinate','r')
lines = fold_file.readlines()
fold_file.close()

a = []
b = []
c = []
d = []
e = []
f = []
g = []

for line in lines:
    line = line.strip('\n').split(' ')
    fold = line[0]
    v = []
    for i in line[1:]:
        if i != '' and i != ' ':
            v.append(i)
    v_3 = v[0:3]
    v_3 = [float(i) for i in v_3]
    if fold[0] == 'a':
        a.append(v_3)
    elif fold[0] == 'b':
        b.append(v_3)
    elif fold[0] == 'c':
        c.append(v_3)
    elif fold[0] == 'd':
        d.append(v_3)
    elif fold[0] == 'e':
        e.append(v_3)
    elif fold[0] == 'f':
        f.append(v_3)
    elif fold[0] == 'g':
        g.append(v_3)

a = np.transpose(np.array(a))
b = np.transpose(np.array(b))
c = np.transpose(np.array(c))
d = np.transpose(np.array(d))
e = np.transpose(np.array(e))
f = np.transpose(np.array(f))
g = np.transpose(np.array(g))

classes = {
                'a':a,
                'b':b,
                'c':c,
                'd':d,
                'e':e,
                'f':f,
                'g':g
}


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

judge = {}

for i in classes.keys():
    if i in judge:
        ax.scatter(classes[i][0], classes[i][1], classes[i][2], alpha=0.8, c=class1[i])
    else:
        ax.scatter(classes[i][0], classes[i][1], classes[i][2], alpha=0.8, c=class1[i], label=label[i])
        judge[i]=1

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.legend(loc="upper left")

plt.show()
plt.savefig('Representation.png')
