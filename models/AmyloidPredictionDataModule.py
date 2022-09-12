
import torch
from torch.utils.data import random_split, DataLoader
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import os
import logging


class AmyloidPredictionDataModule( pl.LightningDataModule ):
    
    def __init__(self, batch_size, train_val_ratio, root_path ):
        
        super().__init__()
         
        self.root_path = root_path
        
        # TODO: change this (?) 
        # reading in csv with subj info and paths
        #logging.info("Reading.. mprage_pib_paths.csv")
        self.df = pd.read_csv( "../data/mprage_pib_paths.csv" )
        
        #logging.info("Reading subject list " + self.df.PTID)
        self.subjectIDs = self.df.PTID 
        
        self.batch_size = batch_size
        
        self.train_val_ratio = train_val_ratio 
        
        self.subjects = None
        
        self.test_subjects = None
        
        self.preprocess = None
        self.transform = None
        
        self.train_set = None
        self.val_set = None
        self.test_set = None
        
    def get_max_shape( self, subjects ):
        
        import numpy as np 
        
        # SubjectsDataset is subclass of torch.data.utils.Dataset 
        dataset = tio.SubjectsDataset(subjects)
        
        shapes = np.array( [s.spatial_shape for s in dataset] )
        
        return shapes.max(axis=0)

    # def download_data(self):
        #image_training_paths = ""
        #image_test_paths = ""
        #label_test_paths = ""
        #label_training_paths = ""
        #return image_training_paths, image_test_paths, label_test_paths, label_training_paths
    
    def prepare_data(self):
        
        mr_dict = zip( self.df.PTID, self.df.MPRAGE_FULL_PATH )
        pib_dict = zip( self.df.PTID, self.df.pib_full_path )
        
        self.subjects = [] 
        
        for ptid in self.subjectIDs:
            
            # creating instance of tio.Subject 
            # Image file, can be any format supported by SimpleITK or NiBabel
            # can be dicom_folder
            # can be .nii
            # can be created using PyTorch tensors or NumPy arrays 
            
            #  mr_dict[ ptid ] = ADNIPET/MRI_PIB/ADNI/005_S_0223/MP-RAGE/...
            
            mprage_path = self.root_path + mr_dict[ ptid ]
            pet_path = self.root_path + pib_dict[ ptid ]
             
            subject = tio.Subject(
                
                # TODO: check this
                transforms = None,                
               
                mprage = tio.ScalarImage( mprage_path ),
                pet = tio.LabelMap( pet_path ),
                
                # other attributes can be set here 
                PTID = ptid
            )
            
            self.subjects.append(subject) 
        
        ''' 
        self.test_subjects = [] 
        
        for image_path in image_test_paths:
            subject = tio.Subject( 
                mprage = tio.ScalarImage(image_path),             
            )
            self.test_subjects.append(subject) 
        '''
        
    def setup(self, stage=None):
        
        num_subjects = len(self.subjects)
        
        num_train_subjects = int( round(num_subjects*self.train_val_ratio))
        
        num_val_subjects = num_subjects - num_train_subjects 
        
        splits = num_train_subjects, num_val_subjects 
        
        train_subjects, val_subjects = random_split(self.subjects, splits) 
        
        self.train_set = tio.SubjectsDataset( train_subjects )   
        
        self.val_set = tio.SubjectsDataset( val_subjects )
        
        # TODO: change later - self.test_subjects
        self.test_set = tio.SubjectsDataset( val_subjects ) 
    
    def train_dataloader(self):
        return DataLoader( self.train_set, self.batch_size, num_workers=2)
    
    def val_dataloader(self):
        return DataLoader( self.val_set, self.batch_size, num_workers = 2 )
        
    
    # TODO: change later - self.test_subjects
    def test_dataloader(self):
        return DataLoader( self.val_set, self.batch_size, num_workers=2)
    


    
    
        
        

