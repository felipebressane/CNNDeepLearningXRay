# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 20:35:51 2023

@author: Felipe Bressane 
"""

import os
from UtilPacket import fPredictXRayLung


DIR_DATASET_VAL_NORMAL='C:/UNIRIO/Projects/DataSet/xRay/resized/val/NORMAL/'
DIR_DATASET_VAL_PNEUMONIA='C:/UNIRIO/Projects/DataSet/xRay/resized/val/PNEUMONIA/'
FILE_MODEL='C:/UNIRIO\Projects\CNNDeepLearning\Models/model.GroupModel3.Model1.BatchSize16.20230212.185210.h5'


TP=0
FP=0
TN=0
FN=0

for file_input in os.listdir(DIR_DATASET_VAL_PNEUMONIA):
        
        file_input_path=os.path.join(DIR_DATASET_VAL_PNEUMONIA, file_input)
            
        if os.path.isfile(file_input_path):  
            print(file_input_path)
            predict=fPredictXRayLung(FILE_MODEL,file_input_path,64,64)
                         
            if predict=='Pneumonia':
                print("TP")                
                TP = TP + 1
            else:   
                print("FP")
                FP = FP + 1
                
             
                
for file_input in os.listdir(DIR_DATASET_VAL_NORMAL):
        
        file_input_path=os.path.join(DIR_DATASET_VAL_NORMAL, file_input)
            
        if os.path.isfile(file_input_path):  
            print(file_input_path)
            predict=fPredictXRayLung(FILE_MODEL,file_input_path,64,64)
                         
            if predict=='Normal':
                print("TN")
                TN = TN + 1
            else:
                print("FN")
                FN = FN + 1
                


             
print(TP)            
print(FP)
print(TN)
print(FN)


            
 