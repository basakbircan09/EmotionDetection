import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

face_cascade = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')
for num in range(1,201):
    img = cv.imread("/home/sam02/Desktop/FEI original/neutral/"+str(num)+"-11.jpg")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if faces == ():
        print(num)
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,0,0),1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        sub_face = img[y:y+h, x:x+w]
        sub_face_son=cv.cvtColor(sub_face, cv.COLOR_BGR2RGB)
        plt.imsave("/home/sam02/Desktop/FEI_alligned/neutral/FEI-n3-"+str(num)+".jpg", sub_face_son)
