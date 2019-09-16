#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 22:52:59 2019

@author: zoro
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1,2,3])
y = np.array([1,2,3])
z = np.array([1,2,3])
fig=plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x,y,z,alpha=0.8)
plt.show()