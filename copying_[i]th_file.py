#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
import shutil
import glob
import random
import sys
from tkinter import Tcl

#path = "/media/lab/WDFAT32/Frames_Faces/gray_scale/neutral5"
path = "/home/lab/Desktop/src/data/sadness_gray/train/sadness"
dirs = os.listdir(path)
dirs.sort()

files_alph = Tcl().call('lsort', '-dict', dirs)
files_every_6th = files_alph [::3]

#path_folder = "/media/lab/WDFAT32/Frames_Faces/gray_scale/neutral5"
path_folder = "/home/lab/Desktop/src/data/sadness_gray/train/sadness"
dest_folder = "/home/lab/Desktop/src/data/sadness_gray/test/sadness"

for file in files_every_6th :
    path = path_folder + "/" + file
    dest = dest_folder + "/" + file
    #shutil.copy(path, dest)
    shutil.move(path, dest)


# In[ ]:




