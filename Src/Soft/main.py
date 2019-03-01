# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 21:42:13 2019

@author: perez
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 14:10:26 2019

@author: perez
"""

import os
import torch
import torch.nn as nn
from torch.utils import data
import json


import decoder
from flickr_dataset import FlickrDataset
from training import Trainer
import evaluate
import save_load


#%% _____________________ Data loading _____________________ 

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

path = os.getcwd()
data_folder = '\\processed_data'  

word_map_file = path + data_folder + '\\WORDMAP.json'
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
    
    
#%% _____________________ Preparing data for training _____________________ 
caps_per_image = 3

datasets = {'training' : FlickrDataset(path + data_folder, 'TRAIN', caps_per_image),
            'test'     : FlickrDataset(path + data_folder, 'TEST', caps_per_image)
            }

batch_params = {'batch_size': 16, 'shuffle':True}
data_loader = data.DataLoader(datasets['training'], **batch_params)

#%% _____________________ Initialize model _____________________  

model_size = {'attention_len':512, 
              'embedding_len':256,
              'features_len': 512,
              'lstm_len':512,
              'wordmap_len':len(word_map)
              }
    
decoder1 = decoder.Decoder(**model_size)


#%% _____________________ Training model _____________________

loss_fn = nn.CrossEntropyLoss()

lr, max_epoch = 1E-3, 1000
trainer = Trainer(device, lr, decoder1)
trainer.trainNet(datasets['training'], data_loader, decoder1, loss_fn, max_epoch)

trainer.learning_rate  = 1e-4
trainer.trainNet(datasets['training'], data_loader, decoder1, loss_fn, max_epoch=1000)

trainer.learning_rate  = 5e-5
trainer.trainNet(datasets['training'], data_loader, decoder1, loss_fn, max_epoch=500)

trainer.learning_rate  = 1e-5
trainer.trainNet(datasets['training'], data_loader, decoder1, loss_fn, max_epoch=600)

trainer.learning_rate  = 1e-6
trainer.trainNet(datasets['training'], data_loader, decoder1, loss_fn, max_epoch=1000)

trainer.learning_rate  = 1e-7
trainer.trainNet(datasets['training'], data_loader, decoder1, loss_fn, max_epoch=5000)

#%% _____________________ Evaluating model _____________________

evaluate.evaluateRandomly(datasets['training'], decoder1, word_map, n=5)
evaluate.evaluateRandomly(datasets['test'], decoder1, word_map, n=5)

datasets['test'] = FlickrDataset(path + data_folder, 'TEST', 1)  # Only 1 cap per image, because BLEU computation is slow
BLEU_1_score = evaluate.computeBLEU(datasets['test'], decoder1, word_map, BLEU_idx = 1)

#%% _____________________ Saving model _____________________

save_load.save_all(1, decoder1, trainer)

#%% _____________________ Loading model _____________________

decoder1 = decoder.Decoder(**model_size)
trainer = save_load.load_all(1, decoder1)


