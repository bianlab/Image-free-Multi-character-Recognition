# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 16:22:23 2019

@author: Lenovo
"""


from __future__ import division


import os
import sys
import time


import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt

import torch.nn as nn


import numpy as np
from sys import path 

from torchvision import transforms




m=list(range(256))


Tensor = torch.IntTensor

for number in range(256):
 m[number]=np.random.rand(32,96)



#print(type(m))
#with open("mult.txt",'w') as f:
'''for z in range(256):
  print(z)
  for x in range(32):
   for y in range(96):
     if(m[z][x][y]>0.5): m[z][x][y]=1
     #else: m[z][x][y]=0
     #f.write(str(int(m[z][x][y])))
     #f.write(",")
    #f.write("\n")'''
m = np.asarray(m,'float')

max_num = m.max()
min_num = m.min()

m=(m-min_num)/(max_num-min_num)
max_num = m.max()
min_num = m.min()
print(m)
np.save('mult7.npy',m)

'''m=np.load('./mult.npy')
print(m)'''

'''f=open("train_data3.txt","w")

dataloader = DataLoader(ImageFolder('/home/divinezeng/unn/train/', img_size=64),
                        batch_size=1, shuffle=False, num_workers=8)


for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))
    for number2 in range(12288):
        inter=m[number2]*input_imgs[0].data.numpy()
        for x3 in range(64):
         for y3 in range(64):
          n[number2]=n[number2]+inter[x3][y3]
        f.write(str(int(n[number2])))
        f.write(",")
    f.write("\n")
    for zero in range(12288):
      n[zero]=0
    print(batch_i)
        #print(step,output.shape)
        #plt.imshow(b_x[0][0].data.numpy(),cmap='gray')
        #print(b_x[0][0].data.numpy())
       


f.close()        
#plt.show()
with open("last_data.txt",'w') as f:
 for z2 in range(28):
   f.write(str(n[z2]))
   f.write(",")
   f.write("\n")'''

