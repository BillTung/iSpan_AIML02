import os
import cv2

folder = r'C:\Users\Jenny\Desktop\topic2\Face_cut'


HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))

for person_filename in os.listdir(folder):
    i = 0
    person_filename_path = os.path.join(folder, person_filename)
    print(person_filename)
    if not os.path.isdir(person_filename_path):
        continue
    
    for filename in os.listdir(person_filename_path):
        path = os.path.join(person_filename_path, filename)
        print(filename)
        gbr1 = cv2.imread(path)
        print(path)

        wajah = HaarCascade.detectMultiScale(gbr1, 1.1, 10)

        if len(wajah) > 0:
            x1, y1, width, height = wajah[0]
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
           
            cv2.namedWindow('Face', cv2.WINDOW_NORMAL)
            cv2.rectangle(gbr1, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow('Face', gbr1)
            
            output_folder = f'C:\\Users\\Jenny\\Desktop\\topic2\\{person_filename}'
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            key = cv2.waitKey(0)

            if key == ord('B') or key == ord('b'):
                output_path = os.path.join(output_folder, person_filename + f'{i}.jpg')
                cv2.imwrite(output_path, cv2.imread(path))
                print(f'儲存人臉圖片至：{output_path}')
                i +=1
                
            elif key == ord('D') or key == ord('d'):
                output_path = os.path.join(output_folder, person_filename + f'{i}.jpg')
                cv2.imwrite(output_path, cv2.imread(path))
                print(f'儲存人臉圖片至：{output_path}')
                i +=1
             
            elif key == ord('L') or key == ord('l'):
                output_path = os.path.join(output_folder, person_filename + f'{i}.jpg')
                cv2.imwrite(output_path, cv2.imread(path))
                print(f'儲存人臉圖片至：{output_path}')
                i +=1
                
            elif key == ord('E') or key == ord('e'):
                output_path = os.path.join(output_folder, person_filename + f'{i}.jpg')
                cv2.imwrite(output_path, cv2.imread(path))
                print(f'儲存人臉圖片至：{output_path}')
                i +=1
            
            elif key == ord('Z') or key == ord('z'):
                output_path = os.path.join(output_folder, person_filename + f'{i}.jpg')
                cv2.imwrite(output_path, cv2.imread(path))
                print(f'儲存人臉圖片至：{output_path}')
                i +=1
                
            elif  key == ord('q') or key == ord('Q'):
                  break
                
                
        
        cv2.destroyAllWindows()
