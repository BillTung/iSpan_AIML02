from PIL import Image  # 導入圖像處理庫 PIL
from keras.models import load_model  # 導入Keras模型加載函数
import numpy as np  # 導入NumPy庫，用於數值計算
from numpy import asarray, expand_dims  # 導入NumPy中的數組處理函数
import pickle  # 導入pickle库，用於數據序列化和反序列化
import cv2  # 導入OpenCV庫，用於圖像處理
from keras_facenet import FaceNet  # 導入FaceNet人臉識別模型
from sqlalchemy import create_engine
import pandas as pd
import datetime
from connect_mysql import opensql_read_sheet_todf ,df_to_sql,opensql_read_sheet_todfuser
import os
import time

# 加載人臉檢測器和人臉識別模型
HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
MyFaceNet = FaceNet()
df_user = opensql_read_sheet_todfuser("302_community","302_all_user") # 呼叫一次全住戶資料,存到df_user變數


# 加載已保存的人臉數據庫
myfile = open("datatest1.pkl", "rb")  # 以二進制唯讀模式（"rb"）打開
database = pickle.load(myfile)
myfile.close()
# 設定文件路徑和上次修改時間的初始值
file_path = "./datatest1.pkl"
last_modified = os.path.getmtime(file_path)


# 打開攝像頭
cap = cv2.VideoCapture(0)
while True:
    # 檢查文件的當前修改時間
    current_modified = os.path.getmtime(file_path)
    # 如果文件已經更新，則讀取文件並跳出循環
    if current_modified > last_modified:
        df_user = opensql_read_sheet_todfuser("302_community","302_all_user") # 重新呼叫一次全住戶資料,存到df_user變數
        myfile = open(file_path, "rb")
        database = pickle.load(myfile)
        myfile.close()
        
        print("臉部特徵檔案已更新!")
    # 更新上次修改時間
    last_modified = current_modified
    
    _, gbr1 = cap.read()  # 從攝像頭讀取帧數
    wajah = HaarCascade.detectMultiScale(gbr1, 1.1, 10)  # 檢測人臉,每個矩形框由四個值表示，分別是矩形框的左上角 x 坐標、左上角 y 坐標、寬度和高度。
    
    if len(wajah) > 0:
        for index ,i in enumerate(wajah):
            face_check = []             # 建立一個空串列, 儲存辨識到名單內的人名、戶號、當下時間
            x1, y1, width, height = i  # 獲取人臉區域的坐標和大小
            x1, y1 = abs(x1), abs(y1)  # 座標取絕對值
            x2, y2 = x1 + width, y1 + height  # 計算人臉區域的右下角座標
            gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)  # 將BGR圖像轉換為RGB圖像
            gbr = Image.fromarray(gbr)  # 轉換為PIL圖像 將OpenCV的BGR格式轉換為PIL的RGB格式
            gbr_array = asarray(gbr)  # 將圖像轉換為Numpy數組
            face = gbr_array[y1:y2, x1:x2]  # 提取人臉區域, 放進 face 變數
            face = Image.fromarray(face)  # 轉換PIL圖像
            face = face.resize((160, 160))  # 調整為指定大小 faceNet 模型訓練時, 是採用 160,160 圖像大小
            face = asarray(face)  # 將圖像轉換為NumPy數組
            face = expand_dims(face, axis=0)  # 在第0维上添加一個維度, 擴展為形狀為(1, 高度, 寬度, 通道數)的數組
            signature = MyFaceNet.embeddings(face)  # 對人臉圖像 face 進行特徵提取的操作
            
            min_dist = 100  # 初始化最小距離為 100
            threshold = 2   # 設定閾值,用於篩選已知人臉
            # 在数据库中寻找与特征向量最接近的人脸
            for key, value in database.items():
                dist = np.linalg.norm(value - signature)  # 計算特徵向量之間的歐式距離
                #print(f"距離{dist}")

                if  dist < min_dist:
                    min_dist = dist
                    identity = str(key)
                confidence = float(100 - min_dist)             # 計算辨識準確度
                confirm_perc = str(round(confidence,2)) + '%'  # 構造識別结果字符串, 取到小數第二位 + % 符號

                #----------------------------------------------------------------------------------------------------------------------------    
                if  min_dist < threshold:  # 根据阈值判断是否为已知人脸
                    
                    cv2.rectangle(gbr1, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 繪製綠色框
                    now = datetime.datetime.now()                     # 取得即時的時間 年/月/日/時/分/秒,並格式化時間
                    formatted = now.strftime('%Y-%m-%d %H:%M:%S')
                    
                    df_date = opensql_read_sheet_todf("302_community","user_record_time") #呼叫資料表內的紀錄,存成df_date
                    
                    if df_user.loc[df_user['name'] == identity].size > 0: # 呼叫資料庫內的住戶資料表, 確認是否在名單內
                        # 取出與name 相同的 floor
                        floor = df_user.loc[df_user['name'] == identity,'floor'].item()
                        
                        if df_date.loc[df_date['name'] == identity,'entrytime'].size > 0:                   # 先確認df 資料內人名對應的時間是否有資料
                            face_check = []
                            df_time = df_date.loc[df_date['name'] == identity,'entrytime'].values[-1]       # 取出 df 對應人名的最後一筆時間
                            if  df_time is not None:     # df_time 內是否無資料, 有資料的話條件成立
                                checkin_time = pd.to_datetime(df_time)                                 # 先將資料庫撈出來的時間轉成datetime
                                time_delta = np.datetime64(now) - np.datetime64(checkin_time)          # 現在時間與資料時間均轉成datetime64 , 計算時間差
                                hours = (time_delta.astype('timedelta64[s]').astype(int)) / 3600       # 將時間差轉先換成秒數後 / 3600 , 轉換成小時          
                                #print(f"與上一筆時間差 , {hours}")
                                if  hours > 0.05:                                                      # 如果與上次進入社區時間差3分鐘,條件就成立
                                    face_check.append(identity)                                        # 將人名寫入list
                                    face_check.append(floor)                                           # 將樓層戶號寫入list
                                    face_check.append(formatted)                                       # 將辨識當下時間寫入list
                                    
                                    df_face = pd.DataFrame([face_check], columns=['name','floor','entrytime'])  # 將單筆資料轉成dataframe 
                                    print("Line119 目前偵測到的資料:",df_face)      
                                    
                                    # 呼叫寫入資料庫函數,輸入參數(df名稱, 字串"資料庫成稱, 字串"資料表名稱")
                                    df_to_sql(df_face,"302_community","user_record_time")
                                            
                        else:
                            if identity not in np.array(df_date)[:,0]:                      # 如果辨識到的人名不在df 內
                                face_check2 = []                                            # 建立另一個空list
                                face_check2.append(identity)                                                # 將人名寫入list
                                face_check2.append(floor)                                               # 將班級寫入list
                                face_check2.append(formatted)                                           # 將辨識當下時間寫入list
                                print("line 131",face_check2,face_check)
                                df_face = pd.DataFrame([face_check2], columns=['name','floor','entrytime'])  # 將單筆資料轉成dataframe 
                                print("目前偵測到的資料2:",df_face) 
                                    
                                # 呼叫寫入資料庫函數,輸入參數(df名稱, 字串"資料庫成稱, 字串"資料表名稱")
                                df_to_sql(df_face,"302_community","user_record_time")  
                                
                else:
                    identity = "Unknow"
                    floor = " "
                    confirm_perc = " "
                    cv2.rectangle(gbr1, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 绘制红色框  

            cv2.putText(gbr1, identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)  # 在帧上添加识别结果文字
            cv2.putText(gbr1, floor, (x1 + 50, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)  # 在帧上添加识别结果文字
            cv2.putText(gbr1, confirm_perc, (x1 + 140, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)  # 在帧上添加识别结果文字
            
        cv2.imshow('Face Detect', gbr1)  # 顯示带有人臉框和識別结果的帧
    else:
        cv2.putText(gbr1, "No face detect", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)  # 在帧上添加识别结果文字
        cv2.imshow("Face Detect", gbr1)
        
    k = cv2.waitKey(1) & 0xFF  # 等待按键输入，如果按下ESC键则退出循环
    if k == 27:
        break
    
cv2.destroyAllWindows()  # 关闭窗口
cap.release()  # 释放摄像头资源
