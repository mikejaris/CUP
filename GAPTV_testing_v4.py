# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:25:23 2022

@author: jaris
"""

import sys
sys.path.append('C:/users/jaris/documents/python scripts/')
from create_event import create_event
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix
import time
from GAPTV_Reconstruction_matrix import At_, A_2, A_3, admmdenoise_cacti, gap_denoise
import matplotlib.colors as colors
import cv2
import os

nx=50
nt=50
nyt=nx+nt
# Z = create_event(nx=nx,nt=nt,w0=5,diam=10)
Z=np.load('C:/users/jaris/documents/Coded ultrafast imaging (compressed ultrafast photography)/ionization_front_movie_nt-50_nx-300.npy')
Zm=np.zeros((nt,nx,nx))
for i,z in enumerate(Z): Zm[i]=cv2.resize(z[150:,150:],(nx,nx),interpolation=cv2.INTER_AREA)
Z=Zm
# D = dok_matrix((nx*(nx+nt),nx*(nx+nt)*nt))
def build_TS():
    mval=1
    mask = np.round(np.random.random((nx,nx)))*mval
    # masks= np.zeros((nx,nyt,nt))
    masks= np.zeros((nx*2,nyt,nt))
    # masks = np.zeros((nx*3,nyt,nt))
    t1=time.time()
    TS = dok_matrix((nx*nyt*2,(nx*nyt)*nt))
    # TS = dok_matrix((nx*nyt*3,(nx*nyt)*nt))
    for i in range(nt):
        masks[:nx,i:nx+i,i]=mask
        masks[nx:2*nx,i:nx+i,i]=abs(mask-mval)
        for j in range(nx):
            for k in range(nyt):
                TS[j*nyt+k,(j*nyt+k) + i*(nyt*nx)]=masks[j,k,i]
                TS[nyt*nx+j*nyt+k,(j*nyt+k) + i*(nyt*nx)]=masks[nx+j,k,i]
                
        # vals = np.zeros((nx,nyt))
        # vals[:nx,:nx]=1
        # vals=vals.flatten()
        # ims=np.argwhere(vals>0)
        # masks[2*nx:,:nx,:]=1
        # for i in range(nt):
        #     TS[ims+2*nx*nyt,ims+i+i*(nx*(nx+nt))]=1
            # masks[2*nx:,i:i+nx,i]=1
            
            
    print('Time-shearing matrix took %i seconds to build' %(time.time()-t1))
    return TS, masks

TS,masks=build_TS()


Zm = np.zeros((nt,nx,nyt))
Zs = np.copy(Zm)
Zs[:,:,:nx]=Z
for i in range(nt):
    Zm[i,:nx,i:nx+i]=Z[i]
        
meas = (TS@Zm.reshape((nt,nx*nyt)).flatten())
# meas = meas.reshape((nx*3,nyt))
meas = meas.reshape((nx*2,nyt))


meas=((-1+2**16)*(meas/meas.max())).astype(float)
meas[meas==0]=1
meas=np.expand_dims(meas,2)
        
MAXB = 65535/nt  # real measurement's data range is Cr times less then simulated measment
        
# def A(x): return A_(x,TS,nx,nt)
# def A(x): return A_3(x,TS,nx,nt)
def A(x): return A_2(x,TS,nx,nt)
def At(y): return At_(y, TS,nx,nt)  # transpose of forward model
        


_lambda = 1 # regularization factor, [original set]
accelerate = False # enable accelerated version of GAP
iter_max = 40 # maximum number of iterations
tv_weight = 1  # TV denoising weight
tv_iter_max = 50 # TV denoising maximum number of iterations each
vgaptv = admmdenoise_cacti(meas, masks, A, At, MAXB=MAXB,_lambda=_lambda, accelerate=accelerate, \
                                         iter_max=iter_max,tv_weight=tv_weight,tv_iter_max=tv_iter_max)
            
# imax=np.array([10,20,50,100,200])
# tvmax=np.array([10,20,40,100])
# tvweights=np.array([0.5,1,2])
    
# for tv_iter_max in tvmax:
#     for iter_max in imax:
#         for tv_weight in tvweights:
#             vgaptv = admmdenoise_cacti(meas, masks, A, At, MAXB=MAXB,_lambda=_lambda, accelerate=accelerate, \
#                                         iter_max=iter_max,tv_weight=tv_weight,tv_iter_max=tv_iter_max)
            
#             np.save('C:/users/jaris/documents/Coded ultrafast imaging (compressed ultrafast photography)/GAP_reconstruction_ionization_front_movie_nt-50_nx-300_iter-%i_tviter-%i_tvweight=%f'%(iter_max,tv_iter_max,tv_weight),vgaptv)
# vals = np.zeros((nx,nx+nt))
# vals[:nx,:nx]=1
# vals=vals.flatten()
# ims=np.argwhere(vals>0)

# for i in range(nt):
#     # D[ims,ims+i+i*(nx*(nx+nt))]=1
#     D[ims,ims+i+i*(nx*(nx+nt))]=1
# plt.close('all')
# fig,ax=plt.subplots(ncols=3)
# for i in range(nt):
#     ax[0].cla()
#     ax[1].cla()
#     ax[2].cla()
#     ax[0].imshow(Z[i],vmin=0,vmax=Z.max())
#     ax[1].imshow(vgaptv[i,:,i:nx+i],vmin=0,vmax=vgaptv.max())
#     tmpa=vgaptv[i,:,i:nx+i]
#     tmpb=Z[i]
#     tmpa[tmpa<.001]=0
#     tmpa=tmpa/tmpa.max()
#     # tmpb[tmpb<.001]=0
#     tmpb=tmpb/tmpb.max()
#     tmp=abs(tmpa-tmpb)#/tmpb.sum()
#     ax[2].imshow(tmp+1,norm=colors.LogNorm(vmin=1,vmax=1+2e-1))
#     ax[0].axis("off")
#     ax[1].axis('off')
#     ax[2].axis('off')
#     ax[0].set_title('Ground Truth',fontsize=12)
#     ax[1].set_title('Reconstructed Event',fontsize=12)
#     ax[2].set_title('MSE',fontsize=12)
#     plt.pause(0.25)
    


# gbs=np.array([4,3,2,1])
# gbs=np.array([4])
# savepath='C:/users/jaris/documents/Coded ultrafast imaging (compressed ultrafast photography)/GAP_reconstruction_ionization_front_movie_nt-50_nx-300_02'
# gb=2
# fname='GAP_reconstruction_ionization_front_movie_bottom_half_nt-50_nx-50_iter-200_tviter-50_tvweight=1_v2'
# if not os.path.isdir(fname): os.mkdir(fname)
# plt.close('all')
# # vgaptv=np.load(fname+'.npy')
# fig,ax=plt.subplots(nrows=2,figsize=(8,16))
# # for gb in gbs:
# for i in range(nt):
#     ax[0].cla()
#     ax[1].cla()
#     # ax[2].cla()
#     ax[0].imshow(Z[i], vmin=0, vmax=1, cmap='gray')
#     # ax[1].imshow(vgaptv[i,:,i:nx+i],vmin=0,vmax=1,cmap='gray')
#     tmpa = cv2.GaussianBlur(vgaptv[i, :, i:nx+i], (4*gb+1, 4*gb+1), float(gb),float(gb))
#     ax[1].imshow(tmpa, vmin=0, vmax=vgaptv[0].mean()*2, cmap='gray')
#     # ax[1].imshow(vgaptv[i,:,i:nx+i], vmin=0, vmax=vgaptv[0].mean()*2, cmap='gray')
#     ax[0].axis("off")
#     ax[1].axis('off')
#     # ax[2].axis('off')
#     # ax[0].set_title('Ground Truth',fontsize=12)
#     # ax[1].set_title('Reconstructed Event',fontsize=12)
#     # ax[2].set_title('Gaussian Blurred',fontsize=12)
#     # plt.pause(0.01)
#         # if i ==10:break
#     plt.savefig(fname+'/frame-%i_gbsig-%i.tiff'%(i,gb),dpi=600, format="tiff")
