# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 21:12:37 2023

@author: cfelipe
"""

from UtilPacket import fRsizeImages


DIR_DATASET_TRAIN_NORMAL='C:/UNIRIO/Projects/DataSet/xRay/train/NORMAL/'
DIR_DATASET_TRAIN_PNEUMONIA='C:/UNIRIO/Projects/DataSet/xRay/train/PNEUMONIA/'

DIR_DATASET_TEST_NORMAL='C:/UNIRIO/Projects/DataSet/xRay/test/NORMAL/'
DIR_DATASET_TEST_PNEUMONIA='C:/UNIRIO/Projects/DataSet/xRay/test/PNEUMONIA/'

DIR_DATASET_VAL_NORMAL='C:/UNIRIO/Projects/DataSet/xRay/val/NORMAL/'
DIR_DATASET_VAL_PNEUMONIA='C:/UNIRIO/Projects/DataSet/xRay/val/PNEUMONIA/'


DIR_DATASET_TRAIN_NORMAL_RSIZED='C:/UNIRIO/Projects/DataSet/xRay/resized/train/NORMAL/'
DIR_DATASET_TRAIN_PNEUMONIA_RSIZED='C:/UNIRIO/Projects/DataSet/xRay/resized/train/PNEUMONIA/'

DIR_DATASET_TEST_NORMAL_RSIZED='C:/UNIRIO/Projects/DataSet/xRay/resized/test/NORMAL/'
DIR_DATASET_TEST_PNEUMONIA_RSIZED='C:/UNIRIO/Projects/DataSet/xRay/resized//test/PNEUMONIA/'

DIR_DATASET_VAL_NORMAL_RSIZED='C:/UNIRIO/Projects/DataSet/xRay/resized/val/NORMAL/'
DIR_DATASET_VAL_PNEUMONIA_RSIZED='C:/UNIRIO/Projects/DataSet/xRay/resized/val/PNEUMONIA/'


fRsizeImages(DIR_DATASET_TRAIN_NORMAL,DIR_DATASET_TRAIN_NORMAL_RSIZED,96,96)
fRsizeImages(DIR_DATASET_TRAIN_PNEUMONIA,DIR_DATASET_TRAIN_PNEUMONIA_RSIZED,96,96)


fRsizeImages(DIR_DATASET_TEST_NORMAL,DIR_DATASET_TEST_NORMAL_RSIZED,96,96)
fRsizeImages(DIR_DATASET_TEST_PNEUMONIA,DIR_DATASET_TEST_PNEUMONIA_RSIZED,96,96)


fRsizeImages(DIR_DATASET_VAL_NORMAL,DIR_DATASET_VAL_NORMAL_RSIZED,96,96)
fRsizeImages(DIR_DATASET_VAL_PNEUMONIA,DIR_DATASET_VAL_PNEUMONIA_RSIZED,96,96)

 
