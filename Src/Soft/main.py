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
from coco_dataset import CocoDataset
from training import Trainer
import evaluate
import save_load


#%% _____________________ Data loading _____________________ 

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

data_folder = os.path.join(os.getcwd(),'processed_data','Coco' )

word_map_file = os.path.join(data_folder,'WORDMAP.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
    
    
#%% _____________________ Preparing data for training _____________________ 
caps_per_image = 2

datasets = {'training' : CocoDataset(data_folder, 'TRAIN'),
            'test'     : CocoDataset(data_folder, 'TEST')
            }

batch_params = {'batch_size': 512, 'shuffle':True}
data_loader = data.DataLoader(datasets['training'], **batch_params)

#%% _____________________ Initialize model _____________________  

model_size = {'attention_len':512, 
              'embedding_len':256,
              'features_len': 512,
              'lstm_len':512,
              'wordmap_len':len(word_map)
              }
    
decoder1 = decoder.Decoder(**model_size)

print(len(datasets['training']))

#%% _____________________ Training model _____________________

loss_fn = nn.CrossEntropyLoss()

lr, max_epoch = 1E-3, 100
trainer = Trainer(device, lr, decoder1)
save_load.save_all('initial', decoder1, trainer)
trainer.trainNet(datasets['training'], data_loader, decoder1, loss_fn, word_map,max_epoch=20)

trainer.learning_rate  = 1e-4
trainer.trainNet(datasets['training'], data_loader, decoder1, loss_fn,word_map, max_epoch=20)
save_load.save_all(1, decoder1, trainer)

trainer.learning_rate  = 5e-5
trainer.trainNet(datasets['training'], data_loader, decoder1, loss_fn,word_map, max_epoch=20)
save_load.save_all(2, decoder1, trainer)

trainer.learning_rate  = 1e-5
trainer.trainNet(datasets['training'], data_loader, decoder1, loss_fn,word_map, max_epoch=20)
save_load.save_all(3, decoder1, trainer)

trainer.learning_rate  = 1e-6
trainer.trainNet(datasets['training'], data_loader, decoder1, loss_fn,word_map, max_epoch=20)
save_load.save_all(4, decoder1, trainer)

trainer.learning_rate  = 1e-7
trainer.trainNet(datasets['training'], data_loader, decoder1, loss_fn,word_map, max_epoch=20)
save_load.save_all(5, decoder1, trainer)

#%% _____________________ Evaluating model _____________________

evaluate.evaluateRandomly(datasets['training'], decoder1, word_map, n=5)
evaluate.evaluateRandomly(datasets['test'], decoder1, word_map, n=5)

datasets['test'] = CocoDataset(data_folder, 'TEST')  # Only 1 cap per image, because BLEU computation is slow
BLEU_1_score = evaluate.computeBLEU(datasets['test'], decoder1, word_map, BLEU_idx = 1)

#%% _____________________ Saving model _____________________

save_load.save_all('final', decoder1, trainer)

#%% _____________________ Loading model _____________________

decoder1 = decoder.Decoder(**model_size)
trainer = save_load.load_all(1, decoder1)


