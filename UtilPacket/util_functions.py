# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 19:57:50 2023

@author: cfelipe
"""
import cv2
import os
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import load_model


def fRsizeImages(dirInputImages,dirOutputImages,reSizeWidth,reSizeHeight):
    
    for file_input in os.listdir(dirInputImages):
        
        file_input_path=os.path.join(dirInputImages, file_input)
            
        if os.path.isfile(file_input_path):  
            
            file_output_path=os.path.join(dirOutputImages, file_input)
            
            image = cv2.imread(file_input_path,cv2.IMREAD_UNCHANGED)
            
            newSize = (reSizeWidth,reSizeHeight)
            
            outputResize = cv2.resize(image, newSize)
                        
            cv2.imwrite(file_output_path,outputResize) 
            
            
            
def fNumberFiles(dirFiles):
    
    numberFiles = 0
    
    for file_name in os.listdir(dirFiles):
        
        file_name_path=os.path.join(dirFiles, file_name)
            
        if os.path.isfile(file_name_path):  
            numberFiles = numberFiles+1
        else:            
            for file_name_leve2 in os.listdir(file_name_path):
                
                file_name_path_leve2=os.path.join(file_name_path, file_name_leve2)
                
                if os.path.isfile(file_name_path_leve2):  
                    numberFiles = numberFiles+1
            
            
    return numberFiles           
                  
           
            
def fComonDivisors(firstValue, secondValue):
    divisors=[]
    
    if firstValue < secondValue:
        baseValue = firstValue
    else:
        baseValue = secondValue
          
    for divisor in range(1,baseValue):
        if firstValue % divisor == 0 and secondValue % divisor == 0:  
            divisors.append(divisor)
            
    return divisors



def fGetBestAccuracy(historyTrainning):
    bestAccuracy=[]
    iBestAccurancy=0
    iItenBestAccurancy=0
    
    for iAccuracy in range(0,len(historyTrainning['val_accuracy'])):
        
        if historyTrainning['val_accuracy'][iAccuracy] > iBestAccurancy:
            iBestAccurancy = historyTrainning['val_accuracy'][iAccuracy]
            iItenBestAccurancy = iAccuracy
            
    bestAccuracy.append(historyTrainning['loss'][iItenBestAccurancy])
    bestAccuracy.append(historyTrainning['accuracy'][iItenBestAccurancy])
    bestAccuracy.append(historyTrainning['val_loss'][iItenBestAccurancy])
    bestAccuracy.append(historyTrainning['val_accuracy'][iItenBestAccurancy])
    
    iItenBestAccurancy += 1
            
    bestAccuracy.append(iItenBestAccurancy)
    
    return bestAccuracy
            

def fPredictXRayLung(model,xRayFile,SizeWidth,SizeHeight):
    
    cnnModel = load_model(model)

    img = xRayFile

    testImage = load_img(img, target_size = (SizeWidth,SizeHeight))

    testImage = img_to_array(testImage)
    
    testImage = np.expand_dims(testImage, axis = 0)
    
    result = cnnModel.predict(testImage)
        
    if result[0][0] == 1:        
        prediction = 'Pneumonia'
    else:        
        prediction = 'Normal'

    return prediction



def fReportHistoric(fileReport,historyTrainning,typeDeepLayer,process,cnn,numberEpoch,batchSize,numberLayers,stepsPerEpoch,validationSteps):
    iAccuracy = 0
    iEpoch = 0

    for iAccuracy in range(0,len(historyTrainning['val_accuracy'])):
        loss=historyTrainning['loss'][iAccuracy]
        accuracy=historyTrainning['accuracy'][iAccuracy]
        val_loss=historyTrainning['val_loss'][iAccuracy]
        val_accuracy=historyTrainning['val_accuracy'][iAccuracy]
        
        iEpoch += 1
        
        fileReport.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%0.5f;%0.5f;%0.5f;%0.5f\n" %(typeDeepLayer,process,cnn,numberEpoch,batchSize,numberLayers,stepsPerEpoch,validationSteps,iEpoch,loss,accuracy,val_loss,val_accuracy))
    
    fileReport.flush()
    