# -*- coding: utf-8 -*-

 

import os
from UtilPacket import fPredictXRayLung


DIR_DATASET_VAL_NORMAL='C:/UNIRIO/Projects/DataSet/xRay/resized/val/NORMAL/'
DIR_DATASET_VAL_PNEUMONIA='C:/UNIRIO/Projects/DataSet/xRay/resized/val/PNEUMONIA/'
FILE_MODEL='C:/UNIRIO\Projects\CNNDeepLearning/cnn.h5'




 
        
file_input_path='C:/UNIRIO/Projects/DataSet/xRay/resized/val/PNEUMONIA/person1946_bacteria_4874.JPEG'
            

            
predict=fPredictXRayLung(FILE_MODEL,file_input_path,96,96)
print(predict)
            
#            if predict=='Normal':
#                print("Normal")
#            else:
#                print("Pneumonia")
            
            
print(file_input_path)