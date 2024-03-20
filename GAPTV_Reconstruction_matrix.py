import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
import time
from scipy.sparse import dok_matrix
import cv2

def A_(x, Phi,nx,nt):
    return (Phi@x.reshape((nt,nx*(nx+nt))).ravel()).reshape((nx,nx+nt))
    # return (Phi@x.reshape((30,200*230)).ravel()).reshape((400,230))

def At_(y, Phi,nx,nt):
    return (Phi.T@y.ravel()).reshape((nt,nx,nx+nt))
    # return (Phi.T@y.ravel()).reshape((30,100,230))

def A_2(x, Phi,nx,nt):
    # return (Phi@x.reshape((30,200*230)).ravel()).reshape((200,230))
    return (Phi@x.reshape((nt,nx*(nx+nt))).ravel()).reshape((2*nx,nx+nt))

def A_3(x, Phi,nx,nt):
    # return (Phi@x.reshape((30,200*230)).ravel()).reshape((200,230))
    return (Phi@x.reshape((nt,nx*(nx+nt))).ravel()).reshape((3*nx,nx+nt))


def gap_denoise(y, Phi_sum, A, At, _lambda=1, accelerate=True, 
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

def admmdenoise_cacti(meas, mask, A, At, v0=None, orig=None, 
                      iframe=0, nframe=1, MAXB=1., maskdirection='plain',
                      **args):
    nrow, ncol, nmask = mask.shape
    t1=time.time()
    x_ = np.zeros((nrow,ncol,nmask*nframe), dtype=np.float32)
    for kf in range(nframe):
        print('\n=== GAP-TV Reconstruction coded frame block %2d of %2d ===' %(kf+1 ,nframe))
        meas_k = meas[:,:,kf+iframe]/MAXB
        v0_k = v0
        mask_sum = np.sum(mask**2, axis=2)
        mask_sum[mask_sum==0] = 1
        x =  gap_denoise(meas_k, mask_sum, A, At, \
                                          x0=v0_k, X_orig=None, **args)
        # x_[:,:,kf*nmask:(kf+1)*nmask] = x_k
        print('Reconstruction time: %0.1f minutes' %((time.time()-t1)/60))
    return x
