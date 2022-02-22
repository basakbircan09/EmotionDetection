from types import AsyncGeneratorType
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
print(tf.version.VERSION)
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from keras.preprocessing import image



file_path="C:/Users/Nilsu Bora/Desktop/Star/extralayered/Surprised_vs_AnSaNe_model.h5"
my_model=load_model(file_path)

#emotion_dict = { 0: "neutral", 1: "sad", 2: "angry", 3: "surprised" , 4:"happy"} #happy
emotion_dict = { 0: "neutral", 1: "sad", 2: "angry", 3: "surprised" }              #surp
#emotion_dict = { 0: "neutral", 1: "sad",  2: "angry" }                             #angry
#emotion_dict = { 0: "neutral", 1: "sad" }                                           #sad

def prepare(filepath):
    IMG_SIZE = 128  
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  
img="C:/Users/Nilsu Bora/Desktop/Star/new_data_haar100/"
cm=np.zeros((2,2))
ipe=100
model_name=file_path[28:]
print("Model Name:",model_name) 
print("\nTest Image Per Emotion:", ipe)


selected_emotion = "surprised"

the_emotion_cm = 0
others_cm= 0
for i in emotion_dict:
    emotion=emotion_dict[i]
    the_emotion=0
    others=0
    for num in range(1,ipe+1):
        num=str(num)
        prediction = my_model.predict([prepare(img+str(emotion)+"/"+str(emotion)+"_"+num+'.jpg')]) 
        
        if prediction == 1:
            the_emotion+=1
        else:
            others+=1

    if emotion == selected_emotion:
        cm[0][0]=the_emotion
        cm[1][0]=others
    else:
        the_emotion_cm += the_emotion
        others_cm+=others
        print(str(emotion))
        print(others)
        print("hgfdskjgh")
        print(others_cm)

cm[0][1]=the_emotion_cm
cm[1][1]=others_cm


FP= float(cm[0][1])
FN= float(cm[1][0])
TP= float(cm[0][0])
TN= float(cm[1][1])

ACC = Accuracy=(TP+TN)/(TP+TN+FP+FN)
"""
MISC = MisclassificationRate = 1 - Accuracy
PREC = Precision = TP/(FP+TP)
REC = Recall = TP/(TP+FN)
SPEC = Specificity = TN/(TN+FP)
F1 = F1_score = 2*Precision*Recall/(Precision+Recall)

print("FP= "+ str(FP))
print("FN= "+ str(FN))
print("TP= "+ str(TP))
print("TN= "+ str(TN))

print("Accuracy= "+ str(ACC))
print("MisclassificationRate= "+ str(MISC))
print("Precision= "+ str(PREC))
print("Recall= "+ str(REC))
print("Specificity= "+ str(SPEC))
print("F1_Score= "+ str(F1))
"""
cm_df = pd.DataFrame(cm,index = [selected_emotion,"Others"], 
                     columns = [selected_emotion, "Others"])
plt.figure(figsize=(6,4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix with Accuracy' + str(ACC))
plt.ylabel('Predicted Values')
plt.xlabel('Actual Values')
plt.rcParams.update({'font.size': 20})
plt.show()

print(cm[0][1])

"""
#perfect = F1_score = 1

results=np.array([[ACC],[MISC],[PREC],[REC],[SPEC],[F1]])

pd.options.display.float_format = '{:,.2f}'.format

df = pd.DataFrame(results,
                     index = ["Overall Accuracy","a","a","a","a","a"], 
                     columns = ['Happy'])
df

print("overall accuracy: " + str(ACC))
print(type(ACC))
print(type([ACC]))

"""

"""
FP = cm.sum(axis=0) - np.diag(cm)  
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)


results=np.array([TPR,TNR,PPV,NPV,FPR,FNR,FDR,ACC])

pd.options.display.float_format = '{:,.2f}'.format

df = pd.DataFrame(results,
                     index = ['RECALL - True positive rate','True negative rate','PRECİSİON - Positive predictive value','Negative predictive value','False positive rate', "False negative rate", "False discovery rate", "Overall accuracy"], 
                     columns = ['Happy','Others'])
df

"""