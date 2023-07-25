import os
from os import listdir
from PIL import Image
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from keras.models import load_model
import numpy as np
import time

import pickle
import cv2
from keras_facenet import FaceNet

HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
MyFaceNet = FaceNet()
# 開始計時
start_time = time.time()
folder='C:/workspace_facenet/phototrain2'
database = {}

# 遍歷人物資料夾
for person_filename in listdir(folder):

    person_filename_path = os.path.join(folder , person_filename)
    print(person_filename)
    if not os.path.isdir(person_filename_path):
        
        continue
    
    signatures = []
    
    #遍歷每個人的照片
    for filename in listdir(person_filename_path):
        path = os.path.join(person_filename_path, filename)
        print(filename)
        gbr1 = cv2.imread(path)
        wajah = HaarCascade.detectMultiScale(gbr1,1.1,10)
    
        if len(wajah)>0:
            x1, y1, width, height = wajah[0]         
        else:
            x1, y1, width, height = 1, 1, 10, 10
            
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        
        gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
        gbr = Image.fromarray(gbr)                  # konversi dari OpenCV ke PIL
        gbr_array = asarray(gbr)
        
        face = gbr_array[y1:y2, x1:x2]                        
        
        face = Image.fromarray(face)                       
        face = face.resize((160,160))
        face = asarray(face)
        face = expand_dims(face, axis=0)
        signature = MyFaceNet.embeddings(face)
        
        signatures.append(signature)
    print('寫入完畢!')
    end_time = time.time()
    #將人物的簽名列表添加到數據庫中
    database[person_filename] = signatures
    # 計算總執行時間
    execution_time = end_time - start_time
    # 輸出執行時間
    print("程式執行時間：", execution_time, "秒")
    #database[os.path.splitext(filename)[0]]=signature
print("Feature Extraction completed successfully!")  
myfile = open("datatest1.pkl", "wb")
pickle.dump(database, myfile)
myfile.close()
    
#print(database)
