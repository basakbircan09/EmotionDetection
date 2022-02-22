#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
 
train_dir = cv2.imread('/home/lab/Desktop/fer2013/happiness/PrivateTest_95094.jpg', cv2.IMREAD_UNCHANGED)
 
width = 128
height = 128
dim = (width, height)
print('Resized Dimensions : ',train_dir.shape)
 # resize image
train_dir = cv2.resize(train_dir, dim)
print('Resized Dimensions : ',train_dir.shape)


# In[ ]:




