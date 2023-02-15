# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 23:27:26 2023

@author: Felipe Bressane 
"""


from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, BatchNormalization
from keras.initializers import Zeros, RandomNormal, glorot_normal, glorot_uniform
from keras.regularizers import l2



class modelosCNN:
    
    ##############################################################
    # Creating a Convolocional Network in a constructor function
    ##############################################################    
    def __init__(self,Width,Height,TypeCNN):
        self.Width = Width
        self.Height = Height
        self.classifier = Sequential()

        if TypeCNN == 1:
            
            self.classifier.add(Conv2D(32, (3, 3), padding="same", input_shape = (self.Width, self.Height, 3), activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
            
            self.classifier.add(Conv2D(48, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
           
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
            
        elif TypeCNN == 2:
            
            self.classifier.add(Conv2D(32, (3, 3), padding="same", input_shape = (self.Width, self.Height, 3), activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
            self.classifier.add(BatchNormalization())
            
            self.classifier.add(Conv2D(48, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
            self.classifier.add(BatchNormalization())
           
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
            self.classifier.add(BatchNormalization())
                        
        elif TypeCNN == 3:
            
            self.classifier.add(Conv2D(32, (3, 3), padding="same", input_shape = (self.Width, self.Height, 3), activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
            self.classifier.add(BatchNormalization())
            self.classifier.add(Dropout(0.25))
            
            self.classifier.add(Conv2D(48, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
            self.classifier.add(BatchNormalization())
            self.classifier.add(Dropout(0.25))
           
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
            self.classifier.add(BatchNormalization())
            self.classifier.add(Dropout(0.25))

        elif TypeCNN == 4:
                
            weight_init = glorot_normal()
                
            self.classifier.add(Conv2D(32, (3, 3), padding="same", kernel_initializer=weight_init, input_shape = (self.Width, self.Height, 3), activation = 'relu'))       
            self.classifier.add(Conv2D(32, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))            
            self.classifier.add(Dropout(0.25))
            
            self.classifier.add(Conv2D(48, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(48, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))            
            self.classifier.add(Dropout(0.25))
           
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))            
            self.classifier.add(Dropout(0.25))       
            
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))            
            self.classifier.add(Dropout(0.25))       

        elif TypeCNN == 5:
                
            weight_init = glorot_normal()
                
            self.classifier.add(Conv2D(32, (3, 3), padding="same", kernel_initializer=weight_init, input_shape = (self.Width, self.Height, 3), activation = 'relu'))       
            self.classifier.add(Conv2D(32, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))   
            self.classifier.add(BatchNormalization())
            self.classifier.add(Dropout(0.25))
            
            self.classifier.add(Conv2D(48, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(48, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))  
            self.classifier.add(BatchNormalization())
            self.classifier.add(Dropout(0.25))
           
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))  
            self.classifier.add(BatchNormalization())
            self.classifier.add(Dropout(0.25))       
            
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))  
            self.classifier.add(BatchNormalization())
            self.classifier.add(Dropout(0.25))   
            
            
        elif TypeCNN == 6:
                                       
            self.classifier.add(Conv2D(32, (3, 3), padding="same", input_shape = (self.Width, self.Height, 3), activation = 'relu'))      
            self.classifier.add(Conv2D(32, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))  
            self.classifier.add(BatchNormalization())
           
            self.classifier.add(Conv2D(48, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(48, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))  
            self.classifier.add(BatchNormalization())
         
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))  
            self.classifier.add(BatchNormalization())
         
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))  
            self.classifier.add(BatchNormalization())

            self.classifier.add(Conv2D(128, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(128, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(128, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))  
            self.classifier.add(BatchNormalization())


        elif TypeCNN == 7:
                
            weight_init = glorot_normal()
                
            self.classifier.add(Conv2D(32, (3, 3), padding="same", kernel_initializer=weight_init, input_shape = (self.Width, self.Height, 3), activation = 'relu'))       
            self.classifier.add(Conv2D(32, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))   
            self.classifier.add(BatchNormalization())        
            
            self.classifier.add(Conv2D(48, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(48, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))  
            self.classifier.add(BatchNormalization())
           
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))  
            self.classifier.add(BatchNormalization())
            
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))  
            self.classifier.add(BatchNormalization())
            
        elif TypeCNN == 8:
                                            
            self.classifier.add(Conv2D(32, (3, 3), padding="same", input_shape = (self.Width, self.Height, 3), activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))                        
            
            self.classifier.add(Conv2D(48, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(48, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
            
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
  
        
        elif TypeCNN == 9:
                                       
           self.classifier.add(Conv2D(32, (3, 3), padding="same", input_shape = (self.Width, self.Height, 3), activation = 'relu'))      
           self.classifier.add(Conv2D(32, (3, 3), padding="same", activation = 'relu'))
           self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
           
           self.classifier.add(Conv2D(48, (3, 3), padding="same", activation = 'relu'))
           self.classifier.add(Conv2D(48, (3, 3), padding="same", activation = 'relu'))
           self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
                     
           self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
           self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
           self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
           self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
            
    
        elif TypeCNN == 10:
                                       
           self.classifier.add(Conv2D(32, (3, 3), padding="same", input_shape = (self.Width, self.Height, 3), activation = 'relu'))      
           self.classifier.add(MaxPooling2D(pool_size = (2, 2))) 
           
           self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
           self.classifier.add(MaxPooling2D(pool_size = (2, 2))) 
           self.classifier.add(Dropout(0.25)) 
                               

        elif TypeCNN == 11:
                                       
           self.classifier.add(Conv2D(32, (3, 3), padding="same", input_shape = (self.Width, self.Height, 3), activation = 'relu'))      
           
           self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
           self.classifier.add(MaxPooling2D(pool_size = (2, 2))) 
           
 
        elif TypeCNN == 12:
            
           self.classifier.add(Conv2D(32, (3, 3), padding="same", input_shape = (self.Width, self.Height, 3), activation = 'relu'))      
           self.classifier.add(Conv2D(32, (3, 3), padding="same", activation = 'relu'))           
           self.classifier.add(MaxPooling2D(pool_size = (2, 2)))  
           
           self.classifier.add(Conv2D(48, (3, 3), padding="same", activation = 'relu'))
           self.classifier.add(Conv2D(48, (3, 3), padding="same", activation = 'relu'))           
           self.classifier.add(MaxPooling2D(pool_size = (2, 2)))  
            
           self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
           self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))
           self.classifier.add(Conv2D(64, (3, 3), padding="same", activation = 'relu'))           
           self.classifier.add(MaxPooling2D(pool_size = (2, 2)))             
                
           self.classifier.add(Conv2D(128, (3, 3), padding="same", activation = 'relu'))
           self.classifier.add(Conv2D(128, (3, 3), padding="same", activation = 'relu'))
           self.classifier.add(Conv2D(128, (3, 3), padding="same", activation = 'relu'))           
           self.classifier.add(MaxPooling2D(pool_size = (2, 2)))                        
           
                        
    def __del__(self):
        del(self.classifier)
          
   
    def getCNN(self):            
       
       self.classifier.add(Flatten())
    
       return self.classifier
   