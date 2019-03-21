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
caps_per_image_training = 3
caps_per_image_test     = 3

datasets = {'training' : FlickrDataset(path + data_folder, 'TRAIN', caps_per_image_training),
            'test'     : FlickrDataset(path + data_folder, 'TEST', caps_per_image_test)
            }

batch_params = {'batch_size': 16, 'shuffle':True}
data_loader = data.DataLoader(datasets['training'], **batch_params)

#%% _____________________ Initialize model _____________________  

model_size = {'nb_annot_vec':16*16,         # Fixed = dimensions of VGG features map
              'attention_len':1024,          # Number of neurons in the attention model
              'embedding_len':100,          # Dimension of the word embedding
              'lstm_len':1024,              # Number of neurons of in hidden state of the LSTM
              'initSize':128,               # Number of neurons in the NN that initialize hidden state of the LSTM
              'wordmap_len':len(word_map)   # Fixed = number of words in the word map
              }

decoder10 = decoder.Decoder(**model_size)


#%% _____________________ Training model _____________________

loss_fn = nn.CrossEntropyLoss()

lr, max_epoch = 1e-2, 2000
trainer = Trainer(device, lr, decoder10)
trainer.trainNet(datasets['training'], data_loader, word_map, decoder10, loss_fn, max_epoch)





trainer.learning_rate  = 1e-3
trainer.trainNet(datasets['training'], data_loader, word_map, decoder10, loss_fn, max_epoch=1000)

trainer.learning_rate  = 1e-4
trainer.trainNet(datasets['training'], data_loader, word_map, decoder10, loss_fn, max_epoch=2000)

trainer.learning_rate  = 5e-5
trainer.trainNet(datasets['training'], data_loader, word_map, decoder9, loss_fn, max_epoch=1000)

# This one 500 iterations at 5e-5 was too much (increased loss)
trainer.learning_rate  = 5e-5
trainer.trainNet(datasets['training'], data_loader, word_map, decoder9, loss_fn, max_epoch=500)

trainer.learning_rate  = 1e-5
trainer.trainNet(datasets['training'], data_loader, word_map, decoder9, loss_fn, max_epoch=1000)

# Final iterations didnt really decreased the loss either
trainer.learning_rate  = 1e-6
trainer.trainNet(datasets['training'], data_loader, word_map, decoder9, loss_fn, max_epoch=500)

#%% _____________________ Evaluating model _____________________

evaluate.evaluateRandomly(datasets['training'], decoder10, word_map, n=5)
evaluate.evaluateRandomly(datasets['test'], decoder10, word_map, n=5)

datasets['test'] = FlickrDataset(path + data_folder, 'TEST', 1)  # Only 1 cap per image, because BLEU computation is slow
BLEU_1_score = evaluate.computeBLEU(datasets['test'], decoder10, word_map, BLEU_idx = 1)

#%% _____________________ Saving model _____________________

save_load.save_all(10, decoder10, trainer)
 
    
#%% _____________________ Loading model _____________________

model_size_load = {'nb_annot_vec':16*16,         # Fixed = dimensions of VGG features map
              'attention_len':1024,          # Number of neurons in the attention model
              'embedding_len':100,          # Dimension of the word embedding
              'lstm_len':1024,               # Number of neurons of in hidden state of the LSTM
              'initSize':128,               # Number of neurons in the NN that initialize hidden state of the LSTM
              'wordmap_len':len(word_map)   # Fixed = number of words in the word map
              }
decoder10 = decoder.Decoder(**model_size_load)
trainer2 = save_load.load_all(10, decoder10)
#trainer.decoder_optimizer.load_state_dict(trainer2.decoder_optimizer.state_dict())

