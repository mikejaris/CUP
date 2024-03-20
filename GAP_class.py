# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 17:30:20 2024

@author: jaris
"""
from dataclasses import dataclass
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from scipy.sparse import dok_matrix

class MatrixMath:
    @classmethod
    def A(cls,x,TS,nx,nt,ndim):
        return (TS@x.reshape((nt,nx*(nx+nt))).ravel()).reshape((ndim*nx,nx+nt))

    @classmethod    
    def At(cls,y,TS,nx,nt):
        return (TS.T@y.ravel()).reshape((nt,nx,nx+nt))
    
    @classmethod
    def Mask(cls,nt,nx=None,ndim=2,mask=None):
        nx = mask.shape[0] if nx is None else nx
        nyt = nx+nt
        mask = np.round(np.random.random((nx,nx))) if mask is None else mask
        masks= np.zeros((nx*ndim,nyt,nt))
        TS = dok_matrix((nx*nyt*ndim,(nx*nyt)*nt))
        for i in range(nt):
            masks[:nx,i:nx+i,i]=mask
            if ndim>1: 
                masks[nx:2*nx,i:nx+i,i]=abs(mask-1)
            for j in range(nx):
                for k in range(nyt):
                    TS[j*nyt+k,(j*nyt+k) + i*(nyt*nx)]=masks[j,k,i]
                    if ndim>1:
                        TS[nyt*nx+j*nyt+k,(j*nyt+k) + i*(nyt*nx)]=masks[nx+j,k,i]
        return TS, masks


@dataclass
class Params:
    lam: float = 1 # regularization factor, [original set]
    accelerate: bool = False # enable accelerated version of GAP
    iter_max: int = 200 # maximum number of iterations
    tv_weight: float = 1  # TV denoising weight
    tv_iter_max: int = 50 # TV denoising maximum number of iterations each


class Reconstruct(Params,MatrixMath):
    def A(x): 
        return MatrixMath.A(x,TS,nx,nt)
    
    def At(y): 
        return MatrixMath.At(y,TS,nx,nt)  # transpose of forward model
    
    
    @classmethod
    def admmdenoise_cacti(cls,meas, mask):
        nrow, ncol, nmask = mask.shape
        mask_sum = np.sum(mask**2, axis=2)
        mask_sum[mask_sum==0] = 1
        x =  cls.gap_denoise(meas, mask_sum,x0=None)
        return x
    
    @classmethod
    def gap_denoise(cls,y, mask_sum, A, At, _lambda=1, accelerate=True, 
                    denoiser='tv', iter_max=50, noise_estimate=False, sigma=None, 
                    tv_weight=0.1, tv_iter_max=5, multichannel=False, x0=None, 
                    X_orig=None, model=None, show_iqa=True):

        # [0] initialization
        if x0 is None:
            x0 = At(y) # default start point (initialized value)
        if not isinstance(sigma, list):
            sigma = [sigma]
        if not isinstance(iter_max, list):
            iter_max = [iter_max] * len(sigma)
        y1=np.zeros(x0.shape)#(nt,nx,nyt)
        v=x0#(nt,nx,nyt)
        # [1] start iteration for reconstruction
        x = x0 # initialization
        val=10
        for idx, nsig in enumerate(sigma): # iterate all noise levels
            for it in range(iter_max[idx]):
                if 5*it%iter_max[idx]==0:
                    print('%i runs of %i total complete' %(it,iter_max[idx]))
                
                v1=x+y1#(nt,nx,nyt)
                yb = A(v1)
                if accelerate: # accelerated version of GAP
                    v=v1+ _lambda*(At((y-yb)/(Phi_sum+val)))
                    vy = v-y1
                    x = denoise_tv_chambolle(vy.T, tv_weight, n_iter_max=tv_iter_max, 
                                             multichannel=multichannel)
                    x=x.T
                    val*=.998
                    tv_weight*=.999
                    y1=x-vy
                else:
                    v=v1+ _lambda*(At((y-yb)/(Phi_sum)))#(nt,nx,nyt)
                    vy = v-y1#(nt,nx,nyt)
                    x = denoise_tv_chambolle(vy.T, tv_weight, n_iter_max=tv_iter_max, 
                                             multichannel=multichannel)#(nyt,nx,nt)
                    x=x.T#(nt,nx,nyt)
                    y1=x-vy
        return x