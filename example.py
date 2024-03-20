# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 00:30:42 2024

@author: mikej
"""

import sys
sys.path.append(r'C:\Users\mikej\Documents\GitHub\CUP')
from GAP_class import Solver, Params
import numpy as np

nx=50
nt=10
nyt=nx+nt
[x,y]=np.meshgrid(np.arange(nyt),np.arange(nx))

z = np.exp(-0.5*(x-nx/2)**2/(10))+np.exp(-0.5*(y-nx/2)**2/(100))
imgs = [z for i in range(nt)]
imgs=np.array(imgs)

p = Params(lam=2)
s = Solver(p=p)
TS,masks = s.Mask(nt=nt,nx=nx)

meas = (TS@imgs.reshape((nt,nx*nyt)).flatten())
meas = meas.reshape((nx*2,nyt))

tst=s.Reconstruct(meas,TS,masks)