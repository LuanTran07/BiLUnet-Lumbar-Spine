
from __future__ import print_function

import numpy as np
import cv2
import csv

from PIL import Image
from PIL import ImageTk

import tkinter as tk
import tkinter.filedialog as tkFileDialog
from tkinter import filedialog

from imutils import perspective
import imutils
import os

import sub_function
from keras.models import *
from keras.layers import *
import time


from sklearn import preprocessing
#from sklearn.externals import joblib
import joblib
import skimage.filters.rank
import skimage.morphology

def swish(x):
    return (K.sigmoid(x) * x)


my_scaler = joblib.load('scaler2.gz')
modelFile = 'MBNET1000_V3.hdf5'
model = load_model(modelFile, custom_objects={'swish' : swish })

root = tk.Tk()
panelA = None
panelB = None
panelC = None

show_var = tk.IntVar()
data1_ = tk.Label(text = "Show Sematic Segmentation:").place(x=770,y=10)
data1_ = tk.Label(text = "Show Final Evaluation:").place(x=1370, y=10)

showXl1 =10
showYl1 =450   
showX =730
showY =200

def Lumbar_inspection(test, path):
    
    PImin = 34.0
    PImax = 84.0
    LLmin = 31
    LLmax = 79
    PILLmin = -10
    PILLmax = 10
    LDImin = 50
    LDImax = 80
    RLLmin = -14
    RLLmax = 11
    SSmin = 20
    SSmax = 65
    PTmin =5
    PTmax =30
 
    PI =0; LL=0; PImLL =0; L4S1 =0; LDI=0; RLL =0; SS =0; PT =0;
    L1 =0; L2=0; L3 =0; L4 =0; L5=0;
    L1L2=0; L2L3 =0; L3L4 =0; L4L5=0; L5S1=0;
    eval_PI = ' '; eval_LL = ' '; eval_PImLL = ' ';
    eval_LDI = ' '; eval_RLL = ' '; eval_SS = ' '; eval_PT = ' ';
    
    imageOri = test.copy()  
    imgOri = cv2.resize(test, (512, 512)) 
    imgArr = np.array([imgOri])
    imgArr = imgArr.astype('float32')
    imgArr /= 255
    
    start = time.time()        
    #imgs = model.predict(imgArr, batch_size=5, verbose=1)
    predic1, imgs = model.predict(imgArr,batch_size=4)
    
    predic_output = my_scaler.inverse_transform(predic1);
    print(predic_output)
    
    [[PI,LL,PImLL,L4S1,LDI,RLL,SS,PT,L1,L2,L3,L4,L5,L1L2,L2L3,L3L4,L4L5,L5S1]] = predic_output;
    

    for i in range(imgs.shape[0]):
        img = np.zeros((imgs.shape[1], imgs.shape[2], 3), dtype=np.uint8)
        
        #Vertebra = np.zeros((imgs.shape[1], imgs.shape[2], 1), dtype=np.uint8)
        #Sacral = np.zeros((imgs.shape[1], imgs.shape[2], 1), dtype=np.uint8)
        #Hip1 = np.zeros((imgs.shape[1], imgs.shape[2], 1), dtype=np.uint8)
        #Hip2 = np.zeros((imgs.shape[1], imgs.shape[2], 1), dtype=np.uint8)
        
        for k in range(len(img)):
            for j in range(len(img[k])):  # cv2.imwrite也是BGR顺序
                num = np.argmax(imgs[i][k][j])
                if num == 0:
                    continue
                    #img[k][j] = [0, 0, 0] # 0                     
                elif num == 1:
                    img[k][j] = [40, 40, 165]     # 1 Vertebra
                    imgOri[k][j] = [0, 0, 255] 
                    #Vertebra[k][j] = [255]
                elif num == 2:
                    img[k][j] = [0, 255, 0]     # 2 Sacral
                    imgOri[k][j] =  [0, 255, 0] 
                    #Sacral[k][j] = [255]
                elif num == 3:
                    img[k][j] = [255, 0, 0]    #3   hip
                    #Hip1[k][j] = [255]
                    imgOri[k][j] = [255, 0, 0] 
                elif num == 4:
                    img[k][j] = [255,255, 0]
                    imgOri[k][j] = [255,255, 0]
                    #Hip2[k][j] = [255]     
           
    image = cv2.resize(img,(test.shape[1], test.shape[0]), interpolation=cv2.INTER_CUBIC)
    seg = image 
    
    test = ((0.7* test) + (0.3 * seg)).astype("uint8")

    if  ((int(PI) >= PImin) & (int(PI) <= PImax)):  
        cv2.putText(test, "PI = " + str(round(PI,2)) + " (OK)",(showXl1, showYl1), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 255, 0), 5)
        eval_PI = 'OK'
    else: 
        cv2.putText(test, "PI = " + str(round(PI,2)) + " (NG)",(showXl1, showYl1), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 0, 255), 5)
        eval_PI = 'NG'
     
        
    if  ((int(LL) >= LLmin) & (int(LL) <= LLmax)):    
        cv2.putText(test, "LL = " + str(round(LL,2)) + " (OK)",(showXl1, showYl1+300), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 255, 0), 5) 
        eval_LL = 'OK'
    else:
        cv2.putText(test, "LL = " + str(round(LL,2)) + " (NL)",(showXl1, showYl1+300), cv2.FONT_HERSHEY_SIMPLEX,3, (0,0,255), 5) 
        eval_LL = 'NG'     
        
        
    if  ((int(PImLL) >= PILLmin) & (int(PImLL) <= PILLmax)):
        cv2.putText(test, "PI-LL = " + str(round(PImLL,2)) + " (OK)",(showXl1, showYl1+600), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 255, 0), 5) 
        eval_PImLL = 'OK'
    else:
        cv2.putText(test, "PI-LL = " + str(round(PImLL,2)) + " (NG)",(showXl1, showYl1+600), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 0, 255), 5) 
        eval_PImLL = 'NG'
        
        
    cv2.putText(test, "L4S1 = " + str(round(L4S1,2)),(showXl1, showYl1+900), cv2.FONT_HERSHEY_SIMPLEX,3, (0,255, 0), 5) 
    
    
    if  ((int(LDI) >= LDImin) & (int(LDI) <= LDImax)):
        cv2.putText(test, "LDI = " + str(round(LDI,2))+ "%" + " (OK)",(showXl1, showYl1+1200), cv2.FONT_HERSHEY_SIMPLEX,3, (0,255, 0), 5)
        eval_LDI = 'OK'
    else:
        cv2.putText(test, "LDI = " + str(round(LDI,2))+ "%" + " (NG)",(showXl1, showYl1+1200), cv2.FONT_HERSHEY_SIMPLEX,3, (0,0, 255), 5)
        eval_LDI = 'NG'
        
    
    if  ((int(RLL) >= RLLmin) & (int(RLL) <= RLLmax)):  
        cv2.putText(test, "RLL = " + str(round(RLL,2))+ " (OK)",(showXl1, showYl1+1500), cv2.FONT_HERSHEY_SIMPLEX,3, (0,255, 0), 5)
        eval_RLL = 'OK'
    else:
        cv2.putText(test, "RLL = " + str(round(RLL,2))+ " (NG)",(showXl1, showYl1+1500), cv2.FONT_HERSHEY_SIMPLEX,3, (0,0, 255), 5)
        eval_RLL = 'NG'
    
    
    if  ((int(SS) >= SSmin) & (int(SS) <= SSmax)):  
        cv2.putText(test, "SS = " + str(round(SS,2)) + " (OK)",(showXl1, showYl1+1800), cv2.FONT_HERSHEY_SIMPLEX,3, (0,255, 0), 5)
        eval_SS = 'OK'
    else: 
        cv2.putText(test, "SS = " + str(round(SS,2)) + " (NG)",(showXl1, showYl1+1800), cv2.FONT_HERSHEY_SIMPLEX,3, (0,0, 255), 5)
        eval_SS = 'NG'
    
    
    if  ((int(PT) >= PTmin) & (int(PT) <= PTmax)):  
        cv2.putText(test, "PT = " + str(round(PT,2)) +" (OK)",(showXl1, showYl1+2100), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 255, 0), 5)
        eval_PT = 'OK'
    else: 
        cv2.putText(test, "PT = " + str(round(PT,2)) +" (NG)",(showXl1, showYl1+2100), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 0, 255), 5)
        eval_PT = 'NG'
        
    
    cv2.putText(test, "L1 = " + str(round(L1,2)),(int(showX+600), int(showY)), cv2.FONT_HERSHEY_SIMPLEX,3, (0,255, 0), 5)
    cv2.putText(test, "L2 = " + str(round(L2,2)),(int(showX+600), int(showY+200)), cv2.FONT_HERSHEY_SIMPLEX,3, (0,255, 0), 5)
    cv2.putText(test, "L3 = " + str(round(L3,2)),(int(showX+600), int(showY+400)), cv2.FONT_HERSHEY_SIMPLEX,3, (0,255, 0), 5)
    cv2.putText(test, "L4 = " + str(round(L4,2)),(int(showX+600), int(showY+600)), cv2.FONT_HERSHEY_SIMPLEX,3, (0,255, 0), 5)
    cv2.putText(test, "L5 = " + str(round(L5,2)),(int(showX+600), int(showY+800)), cv2.FONT_HERSHEY_SIMPLEX,3, (0,255, 0), 5)   
    
    cv2.putText(test, "L1L2 = " + str(round(abs(L1L2),2)),(int(showX+600), int(showY+1000)), cv2.FONT_HERSHEY_SIMPLEX,3, (0,255, 0), 5)  
    cv2.putText(test, "L2L3 = " + str(round(abs(L2L3),2)),(int(showX+600), int(showY+1200)), cv2.FONT_HERSHEY_SIMPLEX,3, (0,255, 0), 5)  
    cv2.putText(test, "L3L4 = " + str(round(abs(L3L4),2)),(int(showX+600), int(showY+1400)), cv2.FONT_HERSHEY_SIMPLEX,3, (0,255, 0), 5) 
    cv2.putText(test, "L4L5 = " + str(round(abs(L4L5),2)),(int(showX+600), int(showY+1600)), cv2.FONT_HERSHEY_SIMPLEX,3, (0,255, 0), 5) 
    cv2.putText(test, "L5S1 = " + str(round(abs(L5S1),2)),(int(showX+600), int(showY+1800)), cv2.FONT_HERSHEY_SIMPLEX,3, (0,255, 0), 5) 
    
    #test = cv2.resize(imgOri,(test.shape[1], test.shape[0]), interpolation=cv2.INTER_CUBIC)
    test  = cv2.resize(test, (407,682))
    cv2.imwrite( path[:-4]+'_eval.png',test)
    image  = cv2.resize(image, (407,682))
    cv2.imwrite( path[:-4]+'_seg.png',image)   
    
    
    list_result_csv2 = []   
    list_result_csv2.append(['PI',str(round(PI,2)), ' '])
    list_result_csv2.append(['LL',str(round(LL,2)), ' '])
    list_result_csv2.append(['PImLL',str(round(PImLL,2)), ' '])
    list_result_csv2.append(['L4S1',str(round(L4S1,2)), ' '])
    list_result_csv2.append(['LDI',str(round(LDI,2)), ' '])
    list_result_csv2.append(['RLL',str(round(RLL,2)), ' '])
    list_result_csv2.append(['SS',str(round(SS,2)), ' '])
    list_result_csv2.append(['PT',str(round(PT,2)), ' '])
    
    list_result_csv2.append(['L1',str(round(L1,2)), ' '])
    list_result_csv2.append(['L2',str(round(L2,2)), ' '])
    list_result_csv2.append(['L3',str(round(L3,2)), ' '])
    list_result_csv2.append(['L4',str(round(L4,2)), ' '])
    list_result_csv2.append(['L5',str(round(L5,2)), ' '])
    
    list_result_csv2.append(['L1L2',str(round(L1L2,2)), ' '])
    list_result_csv2.append(['L2L3',str(round(L2L3,2)), ' '])
    list_result_csv2.append(['L3L4',str(round(L3L4,2)), ' '])
    list_result_csv2.append(['L4L5',str(round(L4L5,2)), ' '])
    list_result_csv2.append(['L5S1',str(round(L5S1,2)), ' '])
    
    with open(path[:-4]+'_data_M2.csv','w',newline='') as write_file:
        writer = csv.writer(write_file)     
        for i in list_result_csv2:
            writer.writerow(i)
    write_file.close()
    
    return test, seg, imageOri


def select_image():
    # grab a reference to the image panels
    global panelA, panelB, panelC
    #global test, seg, imageOri
    # open a file chooser dialog and allow the user to select an input
    # image
    path = tkFileDialog.askopenfilename()
    # ensure a file path was selected
    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        in_img = cv2.imread(path)
        test, seg, imageOri = Lumbar_inspection(in_img, path)
        
        # convert the images to PIL format...
        image = cv2.resize(imageOri, (600, 950)) 
        test = cv2.resize(test, (550, 950)) 
        seg = cv2.resize(seg, (550, 950)) 
                
        image = Image.fromarray(image)
        seg = Image.fromarray(seg)
        test = Image.fromarray(test)
        
        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)
        test = ImageTk.PhotoImage(test)
        seg = ImageTk.PhotoImage(seg)   
            
        # the first panel will store our original image
        panelA = tk.Label(image=image)
        panelA.image = image
        panelA.place(x = 20, y = 40)
        
        panelB = tk.Label(image=seg)
        panelB.image = seg
        panelB.place(x = 600, y = 40)

        panelC = tk.Label(image=test)
        panelC.image = test
        panelC.place(x = 1140, y = 40)                    
        # otherwise, update the image panels

    
def browse_button():
    filename = filedialog.askdirectory()
    print(filename)
    dirs = os.listdir(filename)
    
    for file in dirs:
        in_image = cv2.imread(os.path.join(filename , file))      
        path = filename + "_results/"+ file
        result, seg, imageOri = Lumbar_inspection(in_image, path)

          
btn = tk.Button(root, text="Select an Image for Lumbar Vertebra Inspection", command = select_image)
btn.place(x = 90, y = 10)

button_browse = tk.Button(root,text="Select a Folder to Test", command=browse_button)
button_browse.place(x = 400, y = 10)

root.geometry('1700x1050')
root.mainloop()

    
    
    
    
    
    
    
    
    
    
    
    
    
    