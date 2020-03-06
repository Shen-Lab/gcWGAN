from __future__ import division
from Bio.SeqUtils.ProtParam import ProteinAnalysis 
import numpy as np
import pickle
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import sys
import math
import pickle
import matplotlib.transforms

myset = "success_sample_0.0001_5_64_1.0_semi_diff_100_c.46"

def calculate_biophysical_prop(myfile):
    ssf = []
    aromaticity = []
    gravy = []
    aaf = {}
    count=0.
    with open(myfile) as f:
         for line in f:
             line = line.strip()
             if "No Successful Sequence." in line:
                return -1
             x = ProteinAnalysis(line) 
             ssf.append(x.secondary_structure_fraction())
             aromaticity.append(x.aromaticity())
             gravy.append(x.gravy())
             aaf_temp = x.get_amino_acids_percent()
             aaf = { k: aaf_temp.get(k, 0) + aaf.get(k, 0) for k in set(aaf_temp) | set(aaf) }
             count += 1.
            
    aaf = { k: round(aaf.get(k, 0)/count,3) for k in set(aaf) }
    ssf = np.asarray(ssf)    
    ssf_mean = np.mean(ssf,axis=0)
    aromaticity = np.asarray(aromaticity)
    aromaticity_mean = np.mean(aromaticity)
    gravy = np.asarray(gravy)
    gravy_mean = np.mean(gravy)
    return ssf,aromaticity,gravy,aaf

ssf,aromaticity,gravy,aaf = calculate_biophysical_prop(myset)
