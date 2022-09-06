
import torch
from torch.utils.data import random_split, DataLoader
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import os
import logging


#TODO: set this
root_dir = ""

class AmyloidPredictionDataModule( pl.LightningDataModule ):
    
    def __init__(self, batch_size, train_val_ratio ):
        
        super().__init__()
        
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
        
        shapes = np.array([s.spatial_shape for s in dataset])
        
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
            subject = tio.Subject(
                
                # is this inefficient (?)
                
                mprage = tio.ScalarImage( mr_dict[ ptid ] ),
                pet = tio.LabelMap( pib_dict[ ptid ] ),
                
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
        
        # self.test_set = tio.SubjectsDataset( self.test_subjects ) 
    
    def train_dataloader(self):
        return DataLoader( self.train_set, self.batch_size, num_workers=2)
    
    def val_dataloader(self):
        return DataLoader( self.val_set, self.batch_size, num_workers = 2 )
        
    
    ''' 
    def test_dataloader(self):
        return DataLoader( self.test_set, self.batch_size, num_workers=2)
    '''


class Model(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer

    def prepare_batch(self, batch):
        return batch['image'][tio.DATA], batch['label'][tio.DATA]

    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self.net(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss

    
    
        
        
