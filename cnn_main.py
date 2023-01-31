# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 22:21:47 2023

@author: Felipe Bressane e Vinicius Simas
"""

from CNNPacket import modelosCNN
from CNNPacket import paramCNN
from UtilPacket import fNumberFiles
from UtilPacket import fComonDivisors
from UtilPacket import fGetBestAccuracy
from UtilPacket import fReportHistoric
from tensorflow import keras
import os
import time
import datetime
import tensorflow as tf
from keras.layers import Dense, Dropout, BatchNormalization

 
# Main Dataset directory and sub-directories with image files
DATASET_HOME = 'C:/UNIRIO/Projects/DataSet/xRay/resized/'
TRAIN_DIR    = os.path.join(DATASET_HOME, 'train')
TEST_DIR     = os.path.join(DATASET_HOME, 'test')

# Calculate number files in directories of train and test
nFilesTrain=fNumberFiles(TRAIN_DIR)
nFilesTest=fNumberFiles(TEST_DIR)

# Output files
FILE_RESUME_OUTPUT   = 'C:/UNIRIO/Projects/CNNDeepLearning/Output/resume.model.CNN.txt'
resumeFileModel1CNN=open(FILE_RESUME_OUTPUT,'w')
resumeFileModel1CNN.write("Process;CNN;Epoch;Batch Size;Camadas Ocultas;Steps_per_Epoch;Validation_Steps;Loss;Accuracy;Val_Loss;Val_Accuracy;Best Epoch;Duration\n")

FILE_HISTORY_OUTPUT   = 'C:/UNIRIO/Projects/CNNDeepLearning/Output/history.model.CNN.txt'
historyFileModel1CNN=open(FILE_HISTORY_OUTPUT,'w')
historyFileModel1CNN.write("Process;CNN;Number of Epocch;Batch Size;Camadas Ocultas;Steps_per_Epoch;Validation_Steps;Epoch;Loss;Accuracy;Val_Loss;Val_Accuracy\n")

inputImageWidth  = 64
inputImageHeight = 64


# Create object from parameters class  
ParametersCNN = paramCNN(True,11)

# Set variable HYPER_PARAMETERS with parameters
HYPER_PARAMETERS=ParametersCNN.getParam()


Number_of_Models=len(HYPER_PARAMETERS)

# Trainning each model of the object ParametersCNN
for iParameters in range(0,Number_of_Models):
    
    startProcess=time.time()
      
    Process=HYPER_PARAMETERS[iParameters]["Process"]
        
    CNNType=HYPER_PARAMETERS[iParameters]["CNN"]
        
    Epoca=HYPER_PARAMETERS[iParameters]["Epoca"]
    
    Initial_Rate=HYPER_PARAMETERS[iParameters]["Initial_Rate"]
                
    NumberLayers=len(HYPER_PARAMETERS[iParameters]["Camadas"])
       
    
    ##########################################################
    # At this point begins the construct of the Network Model
    ##########################################################
    
    # Create Network a Model Object using the function Sequential() in constructor function of
    # the class modelosCNN        
    NewModelCNN = modelosCNN(inputImageWidth,inputImageHeight,CNNType)
    
    # Create Model CNN according parameter CNN                       
    classifier = NewModelCNN.getCNN()
                  
    # Create Deep Learning Network
    for iLayers in range(0,NumberLayers):          
        neuronios=HYPER_PARAMETERS[iParameters]["Camadas"][iLayers]["Neuronios"]
        classifier.add(Dense(units = neuronios, activation = 'relu'))
   
        if HYPER_PARAMETERS[iParameters]["Camadas"][iLayers]["Dropout"]==True:            
            classifier.add(Dropout(0.5))     
   
        if HYPER_PARAMETERS[iParameters]["Camadas"][iLayers]["BatchNormal"]==True:            
            classifier.add(BatchNormalization())
           
    # Create output Layer
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    
    if Initial_Rate != 0:          
        opt = keras.optimizers.Adam(learning_rate=Initial_Rate)
        classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
    else:
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
    IMG_SIZE = (inputImageWidth,inputImageHeight)
    
    # Calculate the BatchSizes            
    if ParametersCNN.isTrainning() == True:        
        ListBatchSizes=fComonDivisors(nFilesTrain,nFilesTest)
        iSizeList=len(ListBatchSizes)
    else:
        ListBatchSizes=[]
        ListBatchSizes.append(HYPER_PARAMETERS[iParameters]["BatchSize"])
        iSizeList=1
        
    print(ListBatchSizes)
    # Training the NETWORK for each BatchSIze
    for iBatchSize in range(0,iSizeList): 
        
        BatchSize = ListBatchSizes[iBatchSize]

        train_dataset = tf.keras.utils.image_dataset_from_directory(TRAIN_DIR,shuffle=True,batch_size=BatchSize,image_size=IMG_SIZE)
        test_dataset  = tf.keras.utils.image_dataset_from_directory(TEST_DIR,shuffle=True,batch_size=BatchSize,image_size=IMG_SIZE)
            
        nStepsPerEpoch=round(nFilesTrain/BatchSize)
        nValidationSteps=round(nFilesTest/BatchSize)
     
        # Fitting the CNN to the images                                               
        resul_classifier=classifier.fit_generator(train_dataset, 
                                                  steps_per_epoch = nStepsPerEpoch, 
                                                  epochs = Epoca, 
                                                  validation_data = test_dataset, 
                                                  validation_steps = nValidationSteps)
                
#        print(resul_classifier.params)
#        print(resul_classifier.history.keys())

        bestAccuracy=fGetBestAccuracy(resul_classifier.history)
        
        loss=round(bestAccuracy[0],5)
        accuracy=round(bestAccuracy[1],5)
        val_loss=round(bestAccuracy[2],5)
        val_accuracy=round(bestAccuracy[3],5)
        best_epoch=round(bestAccuracy[4],5)
 
                                                                                           
        fReportHistoric(historyFileModel1CNN,resul_classifier.history,Process,CNNType,Epoca,BatchSize,NumberLayers,nStepsPerEpoch,nValidationSteps)   
 
        duration=time.time()-startProcess                
        resumeFileModel1CNN.write("%s;%s;%s;%s;%s;%s;%s;%0.5f;%0.5f;%0.5f;%0.5f;%s;%s\n" 
        %(Process,CNNType,Epoca,BatchSize,NumberLayers,nStepsPerEpoch,nValidationSteps,loss,accuracy,val_loss,val_accuracy,best_epoch,str(datetime.timedelta(seconds=duration))))
        resumeFileModel1CNN.flush()
        
    if ParametersCNN.isTrainning() == False:
        classifier.save("cnn.h5")
    
    del NewModelCNN

classifier.save("cnn.h5")    
classifier.summary()   
resumeFileModel1CNN.close()    
historyFileModel1CNN.close()  

 



