import numpy as np
import pandas as pd
import sys
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import sys
import math
import pickle

myfile = sys.argv[1]
#### read data
df = pd.read_csv(myfile,delim_whitespace=True)
res_uniq = df['resi1'].unique()
len_res = len(df['resi1'].unique())

# initialize
hbond_contactmap = np.zeros((len_res,len_res))
hbond_sr_bb_contactmap = np.zeros((len_res,len_res))
hbond_lr_bb_contactmap = np.zeros((len_res,len_res))
hbond_bb_sc_contactmap = np.zeros((len_res,len_res))
hbond_sc_contactmap = np.zeros((len_res,len_res))

# get singleton
df_single = df[df['restype2'] == 'onebody']
hbond_single = df_single.hbond_sr_bb + df_single.hbond_lr_bb + df_single.hbond_bb_sc + df_single.hbond_sc
np.fill_diagonal(hbond_contactmap,hbond_single)
np.fill_diagonal(hbond_sr_bb_contactmap,df_single.hbond_sr_bb)
np.fill_diagonal(hbond_lr_bb_contactmap,df_single.hbond_lr_bb)
np.fill_diagonal(hbond_bb_sc_contactmap,df_single.hbond_bb_sc)
np.fill_diagonal(hbond_sc_contactmap,df_single.hbond_sc)


# get pairwise
for i in range(1,len_res+1):
    df_res = df[ (df['resi1'] == i) & (df['restype2'] != 'onebody')] 
    for j in range(df_res.shape[0]):
        hbond_contactmap[i-1,int(df_res.iloc[j].resi2)-1] = float(df_res.iloc[j].hbond_sr_bb) + float(df_res.iloc[j].hbond_lr_bb) + float(df_res.iloc[j].hbond_bb_sc) + float(df_res.iloc[j].hbond_sc)
        hbond_contactmap[int(df_res.iloc[j].resi2)-1,i-1] = hbond_contactmap[i-1,int(df_res.iloc[j].resi2)-1]

        hbond_sr_bb_contactmap[i-1,int(df_res.iloc[j].resi2)-1] = float(df_res.iloc[j].hbond_sr_bb)
        hbond_sr_bb_contactmap[int(df_res.iloc[j].resi2)-1,i-1] = hbond_sr_bb_contactmap[i-1,int(df_res.iloc[j].resi2)-1]

        hbond_lr_bb_contactmap[i-1,int(df_res.iloc[j].resi2)-1] = float(df_res.iloc[j].hbond_sr_bb) 
        hbond_lr_bb_contactmap[int(df_res.iloc[j].resi2)-1,i-1] = hbond_lr_bb_contactmap[i-1,int(df_res.iloc[j].resi2)-1]

        hbond_bb_sc_contactmap[i-1,int(df_res.iloc[j].resi2)-1] = float(df_res.iloc[j].hbond_sr_bb)
        hbond_bb_sc_contactmap[int(df_res.iloc[j].resi2)-1,i-1] = hbond_bb_sc_contactmap[i-1,int(df_res.iloc[j].resi2)-1]

        hbond_sc_contactmap[i-1,int(df_res.iloc[j].resi2)-1] = float(df_res.iloc[j].hbond_sr_bb)
        hbond_sc_contactmap[int(df_res.iloc[j].resi2)-1,i-1] = hbond_sc_contactmap[i-1,int(df_res.iloc[j].resi2)-1]

### saving
np.savetxt(myfile[:-15]+'_hbond.csv',hbond_contactmap,delimiter=',',fmt='%1.3f')
np.savetxt(myfile[:-15]+'_sr_bb_hbond.csv',hbond_sr_bb_contactmap,delimiter=',',fmt='%1.3f')
np.savetxt(myfile[:-15]+'_lr_bb_hbond.csv',hbond_lr_bb_contactmap,delimiter=',',fmt='%1.3f')
np.savetxt(myfile[:-15]+'_bb_sc_hbond.csv',hbond_bb_sc_contactmap,delimiter=',',fmt='%1.3f')
np.savetxt(myfile[:-15]+'_sc_hbond.csv',hbond_sc_contactmap,delimiter=',',fmt='%1.3f')


### ploting
fig = plt.figure()
plt.imshow(hbond_contactmap)
cb = plt.colorbar()
cb.set_label('Hbond energy (Kcal/Mol)')
plt.xlabel('Residues')
plt.ylabel('Residues')
plt.savefig(myfile[:-15]+'_hbond.eps',format='eps')   # save the figure to file
plt.savefig(myfile[:-15]+'_hbond.pdf',format='png')   # save the figure to file
plt.savefig(myfile[:-15]+'_hbond.png',format='pdf')   # save the figure to file
plt.close()


fig = plt.figure()
plt.imshow(hbond_sr_bb_contactmap)
cb = plt.colorbar()
cb.set_label('Hbond energy (Kcal/Mol)')
plt.xlabel('Residues')
plt.ylabel('Residues')
plt.savefig(myfile[:-15]+'_sr_bb_hbond.eps',format='eps')   # save the figure to file
plt.savefig(myfile[:-15]+'_sr_bb_hbond.pdf',format='png')   # save the figure to file
plt.savefig(myfile[:-15]+'_sr_bb_hbond.png',format='pdf')   # save the figure to file
plt.close()


fig = plt.figure()
plt.imshow(hbond_lr_bb_contactmap)
cb = plt.colorbar()
cb.set_label('Hbond energy (Kcal/Mol)')
plt.xlabel('Residues')
plt.ylabel('Residues')
plt.savefig(myfile[:-15]+'_lr_bb_hbond.eps',format='eps')   # save the figure to file
plt.savefig(myfile[:-15]+'_lr_bb_hbond.pdf',format='png')   # save the figure to file
plt.savefig(myfile[:-15]+'_lr_bb_hbond.png',format='pdf')   # save the figure to file
plt.close()


fig = plt.figure()
plt.imshow(hbond_bb_sc_contactmap)
cb = plt.colorbar()
cb.set_label('Hbond energy (Kcal/Mol)')
plt.xlabel('Residues')
plt.ylabel('Residues')
plt.savefig(myfile[:-15]+'_bb_sc_hbond.eps',format='eps')   # save the figure to file
plt.savefig(myfile[:-15]+'_bb_sc_hbond.pdf',format='png')   # save the figure to file
plt.savefig(myfile[:-15]+'_bb_sc_hbond.png',format='pdf')   # save the figure to file
plt.close()


fig = plt.figure()
plt.imshow(hbond_sc_contactmap)
cb = plt.colorbar()
cb.set_label('Hbond energy (Kcal/Mol)')
plt.xlabel('Residues')
plt.ylabel('Residues')
plt.savefig(myfile[:-15]+'_sc_hbond.eps',format='eps')   # save the figure to file
plt.savefig(myfile[:-15]+'_sc_hbond.pdf',format='png')   # save the figure to file
plt.savefig(myfile[:-15]+'_sc_hbond.png',format='pdf')   # save the figure to file
plt.close()


