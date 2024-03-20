# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 17:30:20 2024

@author: jaris
"""
from dataclasses import dataclass
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from scipy.sparse import dok_matrix

class MatMath:
    @staticmethod
    def A(x,TS,nx,nt,ndim):
        return (TS@x.reshape((nt,nx*(nx+nt))).ravel()).reshape((2*nx,nx+nt))

    @staticmethod    
    def At(y,TS,nx,nt):
        return (TS.T@y.ravel()).reshape((nt,nx,nx+nt))
    
    @staticmethod
    def Mask(nt,nx=None,composite=True,mask=None):
        '''
        Returns the 2-D shearing matrix (TS) and a 3-D matrix (dims=(x,y,t))
        of the shifted mask pattern corresponding to each frame in the
        image sequence
        Inputs:
        nt: int = number of timesteps (e.g., nt = 100 ps/10 ps = 10 frames)
        nx: int = shape of the image frame (e.g., 200x200 pixels)
        composite: bool = whether or not both reflections (+/- 12 deg) are captured from DMD
        '''

        ndim = 2 if composite else 1
        nx = mask.shape[0] if nx is None else nx
        nyt = nx+nt #row length of streak camera image
        mask = np.round(np.random.random((nx,nx))) if mask is None else mask #2-D encoding mask pattern on DMD
        masks= np.zeros((nx*ndim,nyt,nt)) #empty 3-D array for 2-D mask on streaked image for each frame of image sequence

        # 2-D array of masks with column and time dimensions flattened
        ## dok_matrix (dicitonary of keys) is an efficient tool
        ## that allows large 2-D matrix to be stored in RAM for computation
        ## matrix operations using TS are essential to reconstructing large datasets
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


class Solver(Params,MatMath):
    def __init__(self,p=None):
        if p is None:
            p = Params()
        Solver.p = p
        Solver.m=MatMath()
        
    @staticmethod
    def A(x,TS,nx,nt,ndim): 
        #return Solver.m.A(x,TS,nx,nt)
        return (TS@x.reshape((nt,nx*(nx+nt))).ravel()).reshape((ndim*nx,nx+nt))

    @staticmethod
    def At(y,TS,nx,nt): 
        #return Solver.m.At(y,TS,nx,nt)  # transpose of forward model
        return (TS.T@y.ravel()).reshape((nt,nx,nx+nt))
    
    
    @classmethod
    def Reconstruct(cls,meas,TS,masks):
        for k,v in vars(cls.p).items():
            setattr(cls,k,v)

        nx, cls.ny, cls.nt = masks.shape
        cls.ndim = 1 if nx < cls.ny else 2
        cls.nx = nx if nx<cls.ny else nx//2
        mask_sum = np.sum(masks**2, axis=2)
        mask_sum[mask_sum==0] = 1
        cls.mask_sum=mask_sum
        cls.TS=TS
        cls.y=meas
        cls.masks=masks
        # cls(meas,TS,masks,nx,ny,nt,mask_sum)
        
        x =  cls.GAP_denoise()
        return x
    
    @classmethod
    def GAP_denoise(cls,**kwargs):
        #for k,v in kwargs.items():
        #    setattr(cls,k,v)

        # [0] initialization
        #if x0 is None:
        x0 = cls.At(cls.y,cls.TS,cls.nx,cls.nt) # default start point (initialized value)

        y1=np.zeros(x0.shape)#(nt,nx,nyt)
        v=x0#(nt,nx,nyt)

        # [1] start iteration for reconstruction
        # initialization, project streak image to frame sequence using
        ## transpose of time-shearing matrix (At*y=TS.T@y) to create initial frame series
        x = x0

        tv_weight=cls.tv_weight
        for it in range(cls.iter_max):
            v1=x+y1 #(nt,nx,nyt)
            yb = cls.A(v1,cls.TS,cls.nx,cls.nt,cls.ndim) #project frames onto streak camera (TS*x = TS@x)
            if cls.accelerate: # accelerated version of GAP
                v=v1+cls.lam*(cls.At((cls.y-yb)/(cls.mask_sum),cls.TS,cls.nx,cls.nt))
                vy = v-y1
                x = denoise_tv_chambolle(vy.T, cls.tv_weight, max_num_iter=cls.tv_iter_max)
                x=x.T
                #val*=.998 # was added to mask_sum
                tv_weight = 0.999*tv_weight
                y1=x-vy
            else:
                v=v1+cls.lam*(cls.At((cls.y-yb)/(cls.mask_sum),cls.TS,cls.nx,cls.nt))#(nt,nx,nyt)
                vy = v-y1#(nt,nx,nyt)
                x = denoise_tv_chambolle(vy.T, tv_weight, max_num_iter=cls.tv_iter_max)
                x=x.T#(nt,nx,nyt)
                y1=x-vy
        return x


#class CUP(MatMath,Solver,Params):