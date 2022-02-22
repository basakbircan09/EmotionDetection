#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import os

vidcap = cv2.VideoCapture('/media/lab/WDFAT32/Videos/selin_ezgi/selin_ezgi_happiness.mov')
success,image = vidcap.read()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

count = 0
while success:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        justface = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 5)  #Frames (only with faces in) with black rectangle around the face

# Following lines should be used according to the image wanted. 
#If one of the rois or justface is used, change the last variable in function imwrite below. 

        roi_gray = gray[y:y+w, x:x+w]  #Gray image of just the face
        roi_color = image[y:y+h, x:x+w] #Colored image of just the face
#        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)  
#        for (ex, ey, ew, eh) in eyes:
#            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)
    path = '/media/lab/WDFAT32/Frames_Faces/gray_scale/happiness4'
    cv2.imwrite(os.path.join(path ,"selin_ezgi_happiness_frame%d.jpg" % count), roi_gray)
    path1 = '/media/lab/WDFAT32/Frames_Faces/colored/happiness8'
    cv2.imwrite(os.path.join(path1 ,"selin_ezgi_happiness_frame%d.jpg" % count), roi_color)# save frame as JPEG file to the path      
    success,image = vidcap.read()
    
    count += 1
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, count *2 )  #Reads every 2 frame


# In[ ]:





# In[ ]:




