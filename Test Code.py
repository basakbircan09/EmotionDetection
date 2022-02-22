import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import cv2 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
file_path="C:/Users/civan/Desktop/Star/model5_20.h5"
my_model=load_model(file_path)
emotion_dict = {0: "angry", 1: "happy", 2: "neutral", 3: "sad", 4: "surprised"}
def prepare(filepath):
    IMG_SIZE = 128  
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  
img="C:/Users/civan/Desktop/Star/new_data_haar/"
cm=np.zeros((5,5))
ipe=100
model_name=file_path[28:]
print("Model Name:",model_name)
print("\nTest Image Per Emotion:", ipe)
for i in emotion_dict:
    emotion=emotion_dict[i]
    H=0
    A=0
    U=0
    S=0
    N=0
    for num in range(1,ipe+1):
        num=str(num)
        prediction = my_model.predict([prepare(img+str(emotion)+"/"+str(emotion)+"_"+num+'.jpg')]) 
        if emotion_dict[np.argmax(prediction)] == "happy":
            H+=1
        elif emotion_dict[np.argmax(prediction)] == "angry":
            A+=1
        elif emotion_dict[np.argmax(prediction)] == "sad":
            U+=1
        elif emotion_dict[np.argmax(prediction)] == "surprised":
            S+=1
        elif emotion_dict[np.argmax(prediction)] == "neutral":
            N+=1
        
    
    cm[i][0]=A
    cm[i][1]=H
    cm[i][2]=N
    cm[i][3]=U
    cm[i][4]=S

cm_df = pd.DataFrame(cm,index = ['Angry','Happy','Neutral','Sad','Surprised'], 
                     columns = ['Angry','Happy','Neutral','Sad','Surprised'])
plt.figure(figsize=(6,4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.rcParams.update({'font.size': 20})
plt.show()

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
                     columns = ['Angry','Happy','Neutral','Sad','Surprised'])
df
