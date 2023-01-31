# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 00:07:18 2023

@author: Felipe Bressane e Vinicius Simas
"""

class paramCNN:
    
    ######################################## 
    # Defining parameters to be user in CNN
    ######################################## 
    def __init__(self, Trainning, TypeDeepLayer ):    
        self.Trainning = Trainning
        
        if Trainning == True:
            
            if TypeDeepLayer == 1:
                
                self.ParamCNN = [{"Process":1,"CNN":3,"Epoca":15,"Initial_Rate":0,"Camadas":[{"Neuronios":256,"Dropout":False,"BatchNormal":True},
                                                                                             {"Neuronios":256,"Dropout":False,"BatchNormal":True},
                                                                                             {"Neuronios":256,"Dropout":True,"BatchNormal":True}]},
                                 {"Process":1,"CNN":1,"Epoca":10,"Initial_Rate":0,"Camadas":[{"Neuronios":128,"Dropout":False,"BatchNormal":False},
                                                                                             {"Neuronios":128,"Dropout":True,"BatchNormal":True}]},
                                 {"Process":1,"CNN":1,"Epoca":15,"Initial_Rate":0,"Camadas":[{"Neuronios":256,"Dropout":False,"BatchNormal":True},
                                                                                             {"Neuronios":256,"Dropout":True,"BatchNormal":True}]},
                                 {"Process":1,"CNN":1,"Epoca":15,"Initial_Rate":0,"Camadas":[{"Neuronios":256,"Dropout":False,"BatchNormal":False},
                                                                                             {"Neuronios":256,"Dropout":True,"BatchNormal":True}]},
                                 {"Process":1,"CNN":2,"Epoca":5,"Initial_Rate":0,"Camadas":[{"Neuronios":256,"Dropout":False,"BatchNormal":True},
                                                                                            {"Neuronios":256,"Dropout":False,"BatchNormal":True},
                                                                                            {"Neuronios":256,"Dropout":True,"BatchNormal":True}]},
                                 {"Process":1,"CNN":2,"Epoca":10,"Initial_Rate":0,"Camadas":[{"Neuronios":128,"Dropout":False,"BatchNormal":False},
                                                                                             {"Neuronios":128,"Dropout":True,"BatchNormal":True}]},
                                 {"Process":1,"CNN":2,"Epoca":15,"Initial_Rate":0,"Camadas":[{"Neuronios":256,"Dropout":False,"BatchNormal":True},
                                                                                             {"Neuronios":256,"Dropout":True,"BatchNormal":True}]},
                                 {"Process":1,"CNN":2,"Epoca":15,"Initial_Rate":0,"Camadas":[{"Neuronios":256,"Dropout":False,"BatchNormal":False},
                                                                                             {"Neuronios":256,"Dropout":True,"BatchNormal":True}]}]
                                                     
            elif TypeDeepLayer == 2:
                
                self.ParamCNN = [{"Process":2,"CNN":3,"Epoca":10,"Initial_Rate":0,"Camadas":[{"Neuronios":256,"Dropout":False,"BatchNormal":True},
                                                                                             {"Neuronios":256,"Dropout":False,"BatchNormal":True},
                                                                                             {"Neuronios":512,"Dropout":True,"BatchNormal":True}]},
                                 {"Process":2,"CNN":3,"Epoca":10,"Initial_Rate":0,"Camadas":[{"Neuronios":256,"Dropout":False,"BatchNormal":False},
                                                                                             {"Neuronios":512,"Dropout":True,"BatchNormal":True}]},
                                 {"Process":2,"CNN":3,"Epoca":15,"Initial_Rate":0,"Camadas":[{"Neuronios":256,"Dropout":False,"BatchNormal":True},
                                                                                             {"Neuronios":512,"Dropout":True,"BatchNormal":True}]},
                                 {"Process":2,"CNN":3,"Epoca":20,"Initial_Rate":0,"Camadas":[{"Neuronios":256,"Dropout":False,"BatchNormal":False},
                                                                                             {"Neuronios":512,"Dropout":True,"BatchNormal":True}]}]

            elif TypeDeepLayer == 3:                
                
                self.ParamCNN = [{"Process":3,"CNN":1,"Epoca":15,"Initial_Rate":0,"Camadas":[{"Neuronios":128,"Dropout":False,"BatchNormal":False},
                                                                                             {"Neuronios":128,"Dropout":False,"BatchNormal":False}]},
                                 {"Process":3,"CNN":2,"Epoca":15,"Initial_Rate":0,"Camadas":[{"Neuronios":128,"Dropout":False,"BatchNormal":False},
                                                                                             {"Neuronios":128,"Dropout":False,"BatchNormal":False}]},
                                 {"Process":3,"CNN":3,"Epoca":15,"Initial_Rate":0,"Camadas":[{"Neuronios":128,"Dropout":False,"BatchNormal":False},
                                                                                             {"Neuronios":128,"Dropout":False,"BatchNormal":False}]}]
                
            elif TypeDeepLayer == 4:                
                  
                self.ParamCNN = [{"Process":4,"CNN":4,"Epoca":20,"Initial_Rate":0,"Camadas":[{"Neuronios":128,"Dropout":False,"BatchNormal":True},
                                                                                               {"Neuronios":256,"Dropout":True,"BatchNormal":True}]}]
                
            elif TypeDeepLayer == 5:                
                      
                self.ParamCNN = [{"Process":5,"CNN":5,"Epoca":30,"Initial_Rate":0,"Camadas":[{"Neuronios":256,"Dropout":False,"BatchNormal":True},
                                                                                             {"Neuronios":256,"Dropout":False,"BatchNormal":True},
                                                                                             {"Neuronios":256,"Dropout":True,"BatchNormal":True}]}]
                                   
            elif TypeDeepLayer == 6:                
                      
                self.ParamCNN = [{"Process":6,"CNN":6,"Epoca":20,"Initial_Rate":0,"Camadas":[{"Neuronios":256,"Dropout":False,"BatchNormal":True},
                                                                                             {"Neuronios":256,"Dropout":False,"BatchNormal":True},
                                                                                             {"Neuronios":512,"Dropout":True,"BatchNormal":True}]}]

            elif TypeDeepLayer == 7:                
                     
                self.ParamCNN = [{"Process":7,"CNN":7,"Epoca":25,"Initial_Rate":0,"Camadas":[{"Neuronios":512,"Dropout":False,"BatchNormal":True},                                                                          
                                                                                             {"Neuronios":1024,"Dropout":True,"BatchNormal":True}]}]

            elif TypeDeepLayer == 8:                
                  
                self.ParamCNN = [{"Process":8,"CNN":4,"Epoca":20,"Initial_Rate":0,"Camadas":[{"Neuronios":128,"Dropout":False,"BatchNormal":True},                       
                                                                                             {"Neuronios":256,"Dropout":True,"BatchNormal":True}]}]

            elif TypeDeepLayer == 9:                
       
                self.ParamCNN = [{"Process":9,"CNN":8,"Epoca":30,"Initial_Rate":0,"Camadas":[{"Neuronios":128,"Dropout":False,"BatchNormal":True},
                                                                                             {"Neuronios":128,"Dropout":False,"BatchNormal":True},
                                                                                             {"Neuronios":256,"Dropout":True,"BatchNormal":True}]}]

            elif TypeDeepLayer == 10:                
       
                self.ParamCNN = [{"Process":10,"CNN":8,"Epoca":30,"Initial_Rate":0,"Camadas":[{"Neuronios":128,"Dropout":False,"BatchNormal":True},                                                                                             
                                                                                              {"Neuronios":256,"Dropout":True,"BatchNormal":True}]}]
                
            elif TypeDeepLayer == 11:
                
                self.ParamCNN = [{"Process":11,"CNN":4,"Epoca":20,"Initial_Rate":0,"Camadas":[{"Neuronios":128,"Dropout":False,"BatchNormal":False},                                                                                                        
                                                                                              {"Neuronios":256,"Dropout":True,"BatchNormal":True}]}]
                
                                                                                 
        else:
            
            self.ParamCNN = [{"Process":0,"CNN":8,"Epoca":20,"Initial_Rate":0,"BatchSize":16,"Camadas":[{"Neuronios":128,"Dropout":False,"BatchNormal":False},                                                                                                        
                                                                                                        {"Neuronios":256,"Dropout":True,"BatchNormal":True}]}]
        
                                                                                 
    def getParam(self):
        return self.ParamCNN

    
    def isTrainning(self):
        return self.Trainning