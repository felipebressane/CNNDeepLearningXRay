# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 20:35:51 2023

@author: cfelipe
"""

import os
from UtilPacket import fPredictXRayLung


DIR_DATASET_VAL_NORMAL='C:/UNIRIO/Projects/DataSet/xRay/resized/test/NORMAL/'
DIR_DATASET_VAL_PNEUMONIA='C:/UNIRIO/Projects/DataSet/xRay/resized/test/PNEUMONIA/'
FILE_MODEL='C:/UNIRIO/Projects/CNNDeepLearning\Models/CNN.model.20230201.232145.h5'




for file_input in os.listdir(DIR_DATASET_VAL_NORMAL):
        
        file_input_path=os.path.join(DIR_DATASET_VAL_NORMAL, file_input)
            
        if os.path.isfile(file_input_path):  
            
            predict=fPredictXRayLung(FILE_MODEL,file_input_path,96,96)
                         
            if predict=='Normal':
                print("Normal")
            else:
                print("Pneumonia")
            
            

            
 