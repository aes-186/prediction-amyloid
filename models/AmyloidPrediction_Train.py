
import torch
from torch.utils.data import random_split, DataLoader
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import os
import logging
from tqdm import tqdm 

import monai #?? 

import AmyloidPredictionDataModule 
from AmyloidPredictionModel import AmyloidPredictionModel as Model

import time 
from datetime import datetime 

# from unet import Unet 

device = "cuda:0" if torch.cuda.is_available() else "cpu"


# instantiate dataModule 
data = AmyloidPredictionDataModule.AmyloidPredictionDataModule(
    batch_size = 16, # TODO: check this 
    train_val_ratio = 0.8,
    root_path = "/home/wangl15@acct.upmchs.net/"
)

# prepare data 
data.prepare_data() 

# setup data 
data.setup() 

print('Training: ', len(data.train_set))
print('Validation: ', len(data.val_set))
print('Test:      ', len(data.test_set))

# the model 
unet = monai.networks.nets.UNet(
    dimensions=3,
    in_channels=1,
    out_channels=3,
    channels=(8, 16, 32, 64),
    strides=(2, 2, 2),
)

# instantiating the model 
model = Model( 
    net = unet,
    criterion = torch.nn.BCELoss(),
    learning_rate = 1e-2,
    optimizer_class = torch.optim.AdamW #TODO: check this
)

early_stopping = pl.callbacks.early_stopping.EarlyStopping(
    monitor='val_loss',
)

# Trainer 
trainer = pl.Trainer(
    gpus=1 if torch.cuda.is_available() else 0,
    precision=16,
    callbacks=[early_stopping],
)

trainer.logger._default_hp_metric = False

start = datetime.now()

print('Training started at', start)

# trainer.fit - loads in our model and data module 
trainer.fit(model=model, datamodule=data)

print('Training duration:', datetime.now() - start)











