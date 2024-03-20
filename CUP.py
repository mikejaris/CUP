# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 17:30:20 2024

@author: jaris
"""
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from scipy.sparse import dok_matrix
from tqdm import tqdm

class MatMath:
    @staticmethod
    def Create_Mask(nt,nx=None,comp_view=True,mask=None):
        '''
        Returns the 2-D shearing matrix (TS) and a 3-D matrix (dims=(x,y,t))
        of the shifted mask pattern corresponding to each frame in the
        image sequence
        Creates a random mask if only nx is provided and mask is left as NoneType
        
        Inputs:
        nt: int = number of timesteps (e.g., nt = 100 ps/10 ps = 10 frames)
        nx (default = None): int = shape of the image frame (e.g., 200x200 pixels)
        -- if None provided will attempt to use shape of input mask parameter
        comp_view: bool = whether or not both reflections (+/- 12 deg) are captured from DMD
        mask (default = None): array, int = random encoding pattern
        -- if None provided, will create a random mask with shape (nx,nx)
        
        '''

        ndim = 2 if comp_view else 1
        nx = mask.shape[0] if nx is None else nx
        nyt = nx+nt #row length of streak camera image
        mask = np.round(np.random.random((nx,nx))) if mask is None else mask #2-D encoding mask pattern on DMD
        masks= np.zeros((nx*ndim,nyt,nt)) #empty 3-D array for 2-D mask on streaked image for each frame of image sequence

        '''
        2-D array of masks with column and time dimensions flattened
        dok_matrix (dicitonary of keys) is an efficient tool
        that allows large 2-D matrix to be stored in RAM for computation
        matrix operations using TS are essential to reconstructing large datasets
        '''
        TS = dok_matrix((nx*nyt*ndim,(nx*nyt)*nt)) 
        for i in tqdm(range(nt),'Creating masked sampling matrix (TS)'):
            masks[:nx,i:nx+i,i]=mask
            if ndim>1: 
                masks[nx:2*nx,i:nx+i,i]=abs(mask-1)
            for j in range(nx):
                for k in range(nyt):
                    TS[j*nyt+k,(j*nyt+k) + i*(nyt*nx)]=masks[j,k,i]
                    if ndim>1:
                        TS[nyt*nx+j*nyt+k,(j*nyt+k) + i*(nyt*nx)]=masks[nx+j,k,i]
        return TS, masks



class Solver:
    def __init__(self,**kwargs):
        params={'lam':1, # scalar weight applied to the update during each iteration of GAP_denoise
                'accelerate':False, # boolean, reduces TV weight during GAP iterations (final result in fewer runs?)
                'iter_max':40, # number of forward-backward iterations GAP_denoise will perform
                'tv_weight':1, # weight of TV denoiser
                'tv_iter_max':20} # number of TV denoise iterations performed during each iteration of GAP_denoise
        for k,v in kwargs.items():
            params[k]=v
        Solver.params=params
        
    @staticmethod
    def A(x,TS,nx,nt,ndim): 
        '''
        Parameters
        ----------
        x : 3-D array (nt frames with shape (nx,nx))
        TS : 2-D time-shearing matrix
        nx : image dimensions
        nt : number of frames
        ndim : 1 if only one DMD view is captured, 2 if both views are captured
        
        Returns
        -------
        time-sheared view (i.e., forward model)

        '''
        return (TS@x.reshape((nt,nx*(nx+nt))).ravel()).reshape((ndim*nx,nx+nt))

    @staticmethod
    def At(y,TS,nx,nt):
        '''
        Parameters
        ----------
        y : streaked image with shape (nx,nx+nt) or (2*nx,nx+nt) if composite view of DMD captured
        TS : 2-D time-shearing matrix
        nx : image dimensions
        nt : number of frames

        Returns
        -------
        nt frames corresponding to time series (i.e., adjoint model)
        '''
        return (TS.T@y.ravel()).reshape((nt,nx,nx+nt)) # transpose of forward model
    
    @classmethod
    def Reconstruct(cls,meas,TS,mask_sum,comp_view=True,**kwargs):
        '''
        Calls the iterative algorithm reported in [arXiv:1511.03890v1] to reconstruct
        a series of frames from an encoded, streaked image
       
        Parameters
        ----------
        meas : streaked image with shape (nx,nx+nt) or (2*nx,nx+nt) if composite view of DMD captured
        TS : 2-D time-shearing matrix
        mask_sum : 2-D array used to normalize measurement for solver
        comp_view: bool = whether or not both reflections (+/- 12 deg) are captured from DMD
        
        Returns
        -------
        nt frames corresponding to time series
        '''
        
        for k,v in (cls.params).items():
            setattr(cls,k,v)
        cls.ndim=2 if comp_view else 1

        nx,nyt = mask_sum.shape
        cls.nx = nx//2 if cls.ndim==2 else nx
        cls.nt = nyt-cls.nx
        
        cls.mask_sum=mask_sum
        cls.TS=TS
        cls.y=meas
        
        return cls.GAP_denoise()
    
    @classmethod
    def GAP_denoise(cls,**kwargs):
        '''
        Generalized alternating projection (GAP) denoiser
        GAP is a forward-backward solver that uses the adjoint model (At)
        to project the 2-D streak measurement to a 3-D array (nt,nx,nx) that
        is subsequently passed into a total variation (TV) denoiser to identify
        a sparse solution (convex optimization)
        
        Inherits values from the class properties (cls)
        '''
        tv_weight=cls.tv_weight
        x = kwargs['x0'] if 'x0' in kwargs.keys() else cls.At(cls.y,cls.TS,cls.nx,cls.nt) # default start point (initialized value)

        y1=np.zeros(x.shape)#(nt,nx,nyt)
        v=np.copy(x)#(nt,nx,nyt)

        ''' initialization, project streak image to frame sequence using
        transpose of time-shearing matrix (At*y=TS.T@y) to create initial frame series '''
        for it in tqdm(range(cls.iter_max),'Solving for reconstruction'):
            v1=x+y1 #(nt,nx,nyt)
            yb = cls.A(v1,cls.TS,cls.nx,cls.nt,cls.ndim) #project frames onto streak camera (TS*x = TS@x)
            
            if cls.accelerate: # accelerated version of GAP
                v=v1+cls.lam*(cls.At((cls.y-yb)/(cls.mask_sum),cls.TS,cls.nx,cls.nt))
                vy = v-y1
                x = denoise_tv_chambolle(vy.T, cls.tv_weight, max_num_iter=cls.tv_iter_max)
                x=x.T
                tv_weight *= 0.999
                y1=x-vy
            
            else:
                v=v1+cls.lam*(cls.At((cls.y-yb)/(cls.mask_sum),cls.TS,cls.nx,cls.nt))#(nt,nx,nyt)
                vy = v-y1#(nt,nx,nyt)
                x = denoise_tv_chambolle(vy.T, tv_weight, max_num_iter=cls.tv_iter_max)
                x=x.T#(nt,nx,nyt)
                y1=x-vy
        return x

