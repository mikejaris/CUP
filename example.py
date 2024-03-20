# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 00:30:42 2024

@author: mikej
"""

import sys
sys.path.append(r'C:\Users\mikej\Documents\GitHub\CUP')
from CUP import Solver, MatMath
import numpy as np
import matplotlib.pyplot as plt
import cv2

# =============================================================================
# Frame dimensions (nx x nx) and number of frames (nt)
# =============================================================================
nx=50
nt=25
nyt=nx+nt #length of rows of streaked image

# =============================================================================
# load dynamic scene (nt,nx,nx) and simulate streaked image (nt,nx,nyt)
# =============================================================================
Z=np.load('C:/users/jaris/documents/Coded ultrafast imaging (compressed ultrafast photography)/ionization_front_movie_nt-50_nx-300.npy')
Zm=np.zeros((nt,nx,nx))
for i,z in enumerate(Z[::(Z.shape[0]//nt)]): Zm[i]=cv2.resize(z[150:,150:],(nx,nx),interpolation=cv2.INTER_AREA)
Z=Zm
Zm = np.zeros((nt,nx,nyt))
Zs = np.copy(Zm)
Zs[:,:,:nx]=Z
for i in range(nt):
    Zm[i,:nx,i:nx+i]=Z[i]
imgs=Zm

# =============================================================================
# initialize solver parameters and create sampling matrices
# =============================================================================
s = Solver(lam=1,accelerate=False,iter_max=40)
# TS,masks = MatMath.Create_Mask(nt=nt,nx=nx) #create mask for simulation
mask = np.round(np.random.random((nx,nx))) #if mask is known
TS,masks = MatMath.Create_Mask(nt=nt,mask=mask)
mask_sum = np.sum(masks**2,axis=2)
# mask_sum[mask_sum==0]=1

# =============================================================================
# (simulation) project the frames on to the streak camera with the mask encoding pattern
# =============================================================================
meas = s.A(imgs,TS,nx,nt,2) #(TS@imgs.reshape((nt,nx*nyt)).flatten())

# =============================================================================
# digitize the data to 16-bit values and normalize the data
# =============================================================================
meas=((-1+2**16)*(meas/meas.max())).astype(float)
meas[meas==0]=1
MAXB = 65535/nt  # real measurement's data range is Cr times less then simulated measment
meas/=MAXB

# =============================================================================
# call the reconstruction algorithm to solve for the frames
# =============================================================================
tst=s.Reconstruct(meas,TS,mask_sum)

# =============================================================================
# visualize the reconstructed output frames
# =============================================================================
for i,img in enumerate(tst):
    # tmp = img # this is the full streak camera view that is reconstructed
    tmp = tst[i,:,i:nx+i]
    plt.cla()
    plt.imshow(tmp,vmin=tst.min(),vmax=tst.max())
    plt.pause(0.5)