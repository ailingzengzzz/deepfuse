# -*- coding: utf-8 -*-
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#data = np.random.randint(0, 255, size=[40, 40, 40])
path = '/home/alzeng/remote/data/human36/S1/MySegmentsMat/ground_truth_bs/Directions/00026.npz'
npz = np.load(path)
label = npz['label'] #96dimension for gt kpt
pvh = npz['pvh'] #4camera volume index with shape [4,64,64,64]
pvh = pvh.astype(np.uint8)
#pvh = torch.from_numpy(pvh).cuda().float()
#label = np.reshape(label,(-1,3))
m,l,s = np.where(pvh[3]==1) #pvh[i]==1 means just look this ith camera,i=0,1,2,3
print('s',s,'m',m,'l',l)
# x = label[:,0]
# y = label[:,1]
# z = label[:,2]

#   ax = plt.subplot(111, projection='3d')
fig = plt.figure()
ax = Axes3D(fig)
#ax.scatter(x,y,z,c='y')
ax.scatter(m,l,s,c='r')
ax.set_zlabel('Z')
ax.set_ylabel('Y')
ax.set_xlabel('X')

plt.show()