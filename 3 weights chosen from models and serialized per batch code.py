# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:35:32 2020

@author: ckaplan
"""

import glob
import h5py


model_list = glob.glob('*.h5')

#conv1w1 = []
conv1w2 = []
conv1w3 = []

for model_path in model_list:
    h= h5py.File(model_path,'r+')
#    conv1w1.append(h.get("conv2d").get("conv2d").get("kernel:0")[0,2,0,15])
    conv1w2.append(h.get("conv2d").get("conv2d").get("kernel:0")[0,4,0,30])
    conv1w3.append(h.get("conv2d").get("conv2d").get("kernel:0")[0,2,0,5])
    

