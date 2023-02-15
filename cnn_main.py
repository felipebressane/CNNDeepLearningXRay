# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 22:21:47 2023

@author: Felipe Bressane 
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
from keras.layers import Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
 

# Dimensions of image files
inputImageWidth  = 64
inputImageHeight = 64

# Type deep layer will be trainning
groupModel = 10
allBatchesSize=False
 
# Main Dataset directory and sub-directories with image files
DATASET_HOME = 'C:/UNIRIO/Projects/DataSet/xRay/resized/'
TRAIN_DIR    = os.path.join(DATASET_HOME, 'train')
TEST_DIR     = os.path.join(DATASET_HOME, 'test')
 
# Output files (Resume and History)
fileTime = datetime.datetime.now()
FILE_RESUME  = 'C:/UNIRIO/Projects/CNNDeepLearning/Experiments/resume.CNN.GroupModel'+str(groupModel)+'.' + fileTime.strftime("%Y%m%d.%H%M%S") + '.txt'
resumeFileModel1CNN=open(FILE_RESUME,'w')
resumeFileModel1CNN.write("Group Model;Model;CNN;Epoch;Batch Size;Camadas Ocultas;Steps_per_Epoch;Validation_Steps;Loss;Accuracy;Val_Loss;Val_Accuracy;Best Epoch;Duration\n")

FILE_HISTORY   = 'C:/UNIRIO/Projects/CNNDeepLearning/Experiments/history.CNN.GroupModel'+str(groupModel)+'.' + fileTime.strftime("%Y%m%d.%H%M%S") + '.txt'
historyFileModel1CNN=open(FILE_HISTORY,'w')
historyFileModel1CNN.write("Group Model;Model;CNN;Number of Epocch;Batch Size;Camadas Ocultas;Steps_per_Epoch;Validation_Steps;Epoch;Loss;Accuracy;Val_Loss;Val_Accuracy\n")
 
# Calculate number files in directories of train and test
nFilesTrain=fNumberFiles(TRAIN_DIR)
nFilesTest=fNumberFiles(TEST_DIR)

# Create object from parameters class  
ParametersCNN = paramCNN(allBatchesSize,groupModel)

# Set variable HYPER_PARAMETERS with parameters
HYPER_PARAMETERS=ParametersCNN.getParam()

Number_of_Models=len(HYPER_PARAMETERS)

# Trainning each model of the object ParametersCNN
for iModel in range(0,Number_of_Models):
    
    startProcess=time.time()
      
    Model=HYPER_PARAMETERS[iModel]["Model"]
        
    CNNType=HYPER_PARAMETERS[iModel]["CNN"]
        
    Epoch=HYPER_PARAMETERS[iModel]["Epoch"]
    
    Initial_Rate=HYPER_PARAMETERS[iModel]["Initial_Rate"]
                
    NumberLayers=len(HYPER_PARAMETERS[iModel]["Layers"])
       
    
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
         
        neurons=HYPER_PARAMETERS[iModel]["Layers"][iLayers]["Neurons"]
        
        classifier.add(Dense(units = neurons, activation = 'relu'))
   
        if HYPER_PARAMETERS[iModel]["Layers"][iLayers]["Dropout"]==True:            
            classifier.add(Dropout(0.5))     
   
        if HYPER_PARAMETERS[iModel]["Layers"][iLayers]["BatchNormal"]==True:            
            classifier.add(BatchNormalization())            
    # End Deep Learning
           
    # Create output Layer
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    
    # Optimizer
    if Initial_Rate != 0:          
        opt = keras.optimizers.Adam(learning_rate=Initial_Rate)
        classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
    else:
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
    # Image Size
    IMG_SIZE = (inputImageWidth,inputImageHeight)
    
    # Calculate the BatchSizes            
    if ParametersCNN.isAllBatchesSize() == True:        
        ListBatchSizes=fComonDivisors(nFilesTrain,nFilesTest)
        iSizeList=len(ListBatchSizes)
    else:
        ListBatchSizes=[]
        ListBatchSizes.append(HYPER_PARAMETERS[iModel]["BatchSize"])
        iSizeList=1
        
    print(ListBatchSizes)
    # Training the NETWORK for each BatchSIze
    for iBatchSize in range(0,iSizeList): 
        
        BatchSize = ListBatchSizes[iBatchSize]
         
        imageGenerator = ImageDataGenerator(rescale = 1./255,
                                            shear_range = 0.2,
                                            zoom_range  = 0.2,
                                            horizontal_flip = True)
                       
        train_dataset = imageGenerator.flow_from_directory(TRAIN_DIR,
                                                           target_size = IMG_SIZE,
                                                           batch_size = BatchSize,
                                                           class_mode = 'binary')
        
        test_dataset = imageGenerator.flow_from_directory(TEST_DIR,
                                                          target_size = IMG_SIZE,
                                                          batch_size = BatchSize,                                                        
                                                          class_mode = 'binary')
                            
        nStepsPerEpoch=round(nFilesTrain/BatchSize)
        nValidationSteps=round(nFilesTest/BatchSize)
     
        # Fitting the CNN to the images                                               
        resul_classifier=classifier.fit_generator(train_dataset, 
                                                  steps_per_epoch = nStepsPerEpoch, 
                                                  epochs = Epoch, 
                                                  validation_data = test_dataset, 
                                                  validation_steps = nValidationSteps)
                                        
        nModel = iModel + 1
        FILE_H5 = 'C:/UNIRIO/Projects/CNNDeepLearning/Models/model.GroupModel'+str(groupModel) + '.Model'+str(nModel)+'.BatchSize'+str(BatchSize)+'.'+fileTime.strftime("%Y%m%d.%H%M%S") + '.h5'
                                
        classifier.save(FILE_H5)                
             
        bestAccuracy=fGetBestAccuracy(resul_classifier.history)
        
        loss=round(bestAccuracy[0],5)
        accuracy=round(bestAccuracy[1],5)
        val_loss=round(bestAccuracy[2],5)
        val_accuracy=round(bestAccuracy[3],5)
        best_epoch=round(bestAccuracy[4],5)
                                                                                            
        duration=time.time()-startProcess                
 
        # Report resume
        resumeFileModel1CNN.write("%s;%s;%s;%s;%s;%s;%s;%s;%0.5f;%0.5f;%0.5f;%0.5f;%s;%s\n"  %(groupModel,Model,CNNType,Epoch,BatchSize,NumberLayers,nStepsPerEpoch,nValidationSteps,loss,accuracy,val_loss,val_accuracy,best_epoch,str(datetime.timedelta(seconds=duration))))        
        
        # Report history
        fReportHistoric(historyFileModel1CNN,resul_classifier.history,groupModel,Model,CNNType,Epoch,BatchSize,NumberLayers,nStepsPerEpoch,nValidationSteps)                                                 
        classifier.summary()

         
    del NewModelCNN
    del classifier

   
resumeFileModel1CNN.close()    
historyFileModel1CNN.close()  

 




