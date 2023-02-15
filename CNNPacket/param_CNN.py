# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 00:07:18 2023

@author: Felipe Bressane 
"""

class paramCNN:
    
    ######################################## 
    # Defining parameters to be user in CNN
    ######################################## 
    def __init__(self, AllBatchesSize, GroupModel ):  
        
        self.AllBatchesSize = AllBatchesSize
        
        if AllBatchesSize == True:
            
            if GroupModel == 1:
                
                self.ParamCNN = [{"Model":1,"CNN":1,"Epoch":20,"Initial_Rate":0,"Layers":[{"Neurons":128,"Dropout":False,"BatchNormal":True},
                                                                                          {"Neurons":256,"Dropout":True,"BatchNormal":True}]},
                                 {"Model":2,"CNN":1,"Epoch":20,"Initial_Rate":0,"Layers":[{"Neurons":128,"Dropout":False,"BatchNormal":False},
                                                                                          {"Neurons":128,"Dropout":False,"BatchNormal":True},
                                                                                          {"Neurons":256,"Dropout":True,"BatchNormal":True}]},
                                 {"Model":3,"CNN":1,"Epoch":20,"Initial_Rate":0.1,"Layers":[{"Neurons":128,"Dropout":False,"BatchNormal":True},
                                                                                            {"Neurons":256,"Dropout":True,"BatchNormal":True}]},
                                 {"Model":4,"CNN":1,"Epoch":20,"Initial_Rate":0.1,"Layers":[{"Neurons":128,"Dropout":False,"BatchNormal":False},
                                                                                            {"Neurons":256,"Dropout":True,"BatchNormal":True}]}]
                                                                     
            elif GroupModel == 2:
                
               self.ParamCNN = [{"Model":1,"CNN":2,"Epoch":20,"Initial_Rate":0,"Layers":[{"Neurons":128,"Dropout":False,"BatchNormal":False},
                                                                                         {"Neurons":256,"Dropout":True,"BatchNormal":False}]},
                                {"Model":2,"CNN":2,"Epoch":20,"Initial_Rate":0,"Layers":[{"Neurons":128,"Dropout":False,"BatchNormal":False},
                                                                                         {"Neurons":128,"Dropout":False,"BatchNormal":False},
                                                                                         {"Neurons":256,"Dropout":True,"BatchNormal":False}]},
                                {"Model":3,"CNN":2,"Epoch":20,"Initial_Rate":0.1,"Layers":[{"Neurons":128,"Dropout":False,"BatchNormal":False},
                                                                                           {"Neurons":256,"Dropout":True,"BatchNormal":False}]},
                                {"Model":4,"CNN":2,"Epoch":20,"Initial_Rate":0.1,"Layers":[{"Neurons":128,"Dropout":False,"BatchNormal":False},
                                                                                           {"Neurons":256,"Dropout":True,"BatchNormal":False}]}]

            elif GroupModel == 3:                
                
               self.ParamCNN = [{"Model":1,"CNN":3,"Epoch":20,"Initial_Rate":0,"Layers":[{"Neurons":128,"Dropout":False,"BatchNormal":True},
                                                                                         {"Neurons":256,"Dropout":True,"BatchNormal":True}]},
                                {"Model":2,"CNN":3,"Epoch":20,"Initial_Rate":0,"Layers":[{"Neurons":128,"Dropout":False,"BatchNormal":False},
                                                                                         {"Neurons":128,"Dropout":False,"BatchNormal":True},
                                                                                         {"Neurons":256,"Dropout":True,"BatchNormal":True}]},
                                {"Model":3,"CNN":3,"Epoch":20,"Initial_Rate":0.1,"Layers":[{"Neurons":128,"Dropout":False,"BatchNormal":True},
                                                                                           {"Neurons":256,"Dropout":True,"BatchNormal":True}]},
                                {"Model":4,"CNN":3,"Epoch":20,"Initial_Rate":0.1,"Layers":[{"Neurons":128,"Dropout":False,"BatchNormal":False},
                                                                                           {"Neurons":256,"Dropout":True,"BatchNormal":True}]}]

            elif GroupModel == 4:                
                
               self.ParamCNN = [{"Model":1,"CNN":4,"Epoch":20,"Initial_Rate":0,"Layers":[{"Neurons":256,"Dropout":False,"BatchNormal":True},
                                                                                         {"Neurons":512,"Dropout":True,"BatchNormal":True}]},
                                {"Model":2,"CNN":4,"Epoch":20,"Initial_Rate":0,"Layers":[{"Neurons":256,"Dropout":False,"BatchNormal":False},
                                                                                         {"Neurons":256,"Dropout":False,"BatchNormal":True},
                                                                                         {"Neurons":512,"Dropout":True,"BatchNormal":True}]}] 
            elif GroupModel == 5:                
                
               self.ParamCNN = [{"Model":1,"CNN":5,"Epoch":20,"Initial_Rate":0,"Layers":[{"Neurons":256,"Dropout":False,"BatchNormal":True},
                                                                                         {"Neurons":512,"Dropout":True,"BatchNormal":True}]},
                                {"Model":2,"CNN":5,"Epoch":20,"Initial_Rate":0,"Layers":[{"Neurons":256,"Dropout":False,"BatchNormal":False},
                                                                                         {"Neurons":256,"Dropout":False,"BatchNormal":True},
                                                                                         {"Neurons":512,"Dropout":True,"BatchNormal":True}]}] 
            elif GroupModel == 6:                
               
               self.ParamCNN = [{"Model":1,"CNN":6,"Epoch":20,"Initial_Rate":0,"Layers":[{"Neurons":512,"Dropout":False,"BatchNormal":True},
                                                                                         {"Neurons":1024,"Dropout":True,"BatchNormal":True}]}]

            elif GroupModel == 7:                
                
               self.ParamCNN = [{"Model":1,"CNN":7,"Epoch":20,"Initial_Rate":0,"Layers":[{"Neurons":256,"Dropout":False,"BatchNormal":True},
                                                                                         {"Neurons":512,"Dropout":True,"BatchNormal":True}]},
                                {"Model":2,"CNN":7,"Epoch":20,"Initial_Rate":0,"Layers":[{"Neurons":256,"Dropout":False,"BatchNormal":False},
                                                                                         {"Neurons":256,"Dropout":False,"BatchNormal":True},
                                                                                         {"Neurons":512,"Dropout":True,"BatchNormal":True}]}]                
            elif GroupModel == 8:                
               
               self.ParamCNN = [{"Model":1,"CNN":8,"Epoch":20,"Initial_Rate":0,"Layers":[{"Neurons":128,"Dropout":False,"BatchNormal":True},
                                                                                         {"Neurons":256,"Dropout":True,"BatchNormal":True}]}]
                            
            elif GroupModel == 9:                
               
               self.ParamCNN = [{"Model":1,"CNN":9,"Epoch":20,"Initial_Rate":0,"Layers":[{"Neurons":256,"Dropout":False,"BatchNormal":True},
                                                                                           {"Neurons":512,"Dropout":True,"BatchNormal":True}]}]   

            elif GroupModel == 10:                
               
               self.ParamCNN = [{"Model":1,"CNN":10,"Epoch":20,"Initial_Rate":0,"Layers":[{"Neurons":128,"Dropout":False,"BatchNormal":True}]}]
             
                                                                               
            elif GroupModel == 11:                
               
               self.ParamCNN = [{"Model":1,"CNN":11,"Epoch":20,"Initial_Rate":0,"Layers":[{"Neurons":128,"Dropout":False,"BatchNormal":True}]}]


                                                                                                                                
        else:
            
            if GroupModel == 1:
                                
                self.ParamCNN = [{"Model":1,"CNN":1,"Epoch":5,"Initial_Rate":0,"BatchSize":16,"Layers":[{"Neurons":128,"Dropout":False,"BatchNormal":False},
                                                                                                        {"Neurons":256,"Dropout":True,"BatchNormal":True}]}]              
 
            elif GroupModel == 2:
    
                self.ParamCNN = [{"Model":2,"CNN":1,"Epoch":5,"Initial_Rate":0,"BatchSize":16,"Layers":[{"Neurons":512,"Dropout":False,"BatchNormal":False},
                                                                                                        {"Neurons":1024,"Dropout":True,"BatchNormal":True}]}]              
                
            elif GroupModel == 3:
              
                self.ParamCNN = [{"Model":3,"CNN":1,"Epoch":10,"Initial_Rate":0,"BatchSize":16,"Layers":[{"Neurons":512,"Dropout":False,"BatchNormal":False},
                                                                                                        {"Neurons":1024,"Dropout":True,"BatchNormal":True}]}]              
               
            elif GroupModel == 4:
             
                self.ParamCNN = [{"Model":4,"CNN":10,"Epoch":10,"Initial_Rate":0,"BatchSize":16,"Layers":[{"Neurons":512,"Dropout":False,"BatchNormal":False},
                                                                                                         {"Neurons":1024,"Dropout":True,"BatchNormal":True}]}]              

            elif GroupModel == 5:
             
                self.ParamCNN = [{"Model":5,"CNN":10,"Epoch":20,"Initial_Rate":0,"BatchSize":16,"Layers":[{"Neurons":512,"Dropout":False,"BatchNormal":False},
                                                                                                          {"Neurons":1024,"Dropout":True,"BatchNormal":True}]}]              
 
            elif GroupModel == 6:
             
                self.ParamCNN = [{"Model":6,"CNN":10,"Epoch":30,"Initial_Rate":0,"BatchSize":16,"Layers":[{"Neurons":512,"Dropout":False,"BatchNormal":False},
                                                                                                          {"Neurons":1024,"Dropout":True,"BatchNormal":True}]}]              

            elif GroupModel == 7:
                                 
                self.ParamCNN = [{"Model":7,"CNN":1,"Epoch":30,"Initial_Rate":0,"BatchSize":16,"Layers":[{"Neurons":128,"Dropout":False,"BatchNormal":False},
                                                                                                         {"Neurons":128,"Dropout":False,"BatchNormal":True},
                                                                                                         {"Neurons":256,"Dropout":True,"BatchNormal":True}]}]
            elif GroupModel == 8:
                                 
                self.ParamCNN = [{"Model":8,"CNN":1,"Epoch":30,"Initial_Rate":0,"BatchSize":16,"Layers":[{"Neurons":512,"Dropout":False,"BatchNormal":False},
                                                                                                         {"Neurons":512,"Dropout":False,"BatchNormal":True},
                                                                                                         {"Neurons":1024,"Dropout":True,"BatchNormal":True}]}]                
            elif GroupModel == 9:
                                 
                self.ParamCNN = [{"Model":9,"CNN":1,"Epoch":30,"Initial_Rate":0,"BatchSize":16,"Layers":[{"Neurons":512,"Dropout":False,"BatchNormal":False},
                                                                                                         {"Neurons":512,"Dropout":False,"BatchNormal":False},
                                                                                                         {"Neurons":1024,"Dropout":True,"BatchNormal":False}]}]
            elif GroupModel == 10:
                                 
                self.ParamCNN = [{"Model":10,"CNN":12,"Epoch":12,"Initial_Rate":0,"BatchSize":16,"Layers":[{"Neurons":1024,"Dropout":False,"BatchNormal":False},
                                                                                                           {"Neurons":2048,"Dropout":True,"BatchNormal":False}]}]


                                                                                 
    def getParam(self):
        return self.ParamCNN

    
    def isAllBatchesSize(self):
        return self.AllBatchesSize