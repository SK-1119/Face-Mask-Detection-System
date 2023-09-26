# ====================================================================================
#  Author: Kunal SK Sukhija
# ====================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import missingno as msno
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
#%%
model=tf.keras.models.load_model('Face_mask_model.h5')
#%%
cascade=cv2.CascadeClassifier(cv2.data.haarcascades+r"haarcascade_frontalface_default.xml")
#%%
cap=cv2.VideoCapture(0)
while(cap.isOpened()):
    _,frame=cap.read()
    faces=cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=4)
    for x,y,w,h in faces:
        face=frame[y:y+h,x:x+w]
        cv2.imwrite("face.jpg",face)#Save image
        face=tf.keras.preprocessing.image.load_img("face.jpg",
                                                   target_size=(150,150,3))
        face=tf.keras.preprocessing.image.img_to_array(face)#Convert face to numpy array
        face=np.expand_dims(face,axis=0)#Convert to 4D-batch size concept

        ans=model.predict(face)
        if ans<=0.5:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(frame,"With Mask",(x//2,y//2),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))

        else:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(frame,"Without Mask",(x//2,y//2),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))



    cv2.imshow("Frame",frame)
    if cv2.waitKey(100)%256 == 27:
        # cv2.waitKey(1000)
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        # cv2.putText(frame,"Exiting",(x//2,y//2),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
        # cv2.waitKey(1000)
        break
#%%
cap.release()
cv2.destroyAllWindows()
