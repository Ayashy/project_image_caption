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
import save_load as sl


#%% _____________________ Data loading _____________________ 

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

path = os.getcwd()
data_folder = '\\processed_data'  

word_map_file = path + data_folder + '\\WORDMAP.json'
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
    
    
#%% _____________________ Preparing data for training _____________________ 
caps_per_image_training = 4
caps_per_image_test     = 5

datasets = {'training' : FlickrDataset(path + data_folder, 'TRAIN', caps_per_image_training),
            'validation'    : FlickrDataset(path + data_folder, 'EVAL', caps_per_image_test),
            'test'     : FlickrDataset(path + data_folder, 'TEST', caps_per_image_test)
            }

batch_params = {'batch_size': 32, 'shuffle':True}
data_loader = data.DataLoader(datasets['training'], **batch_params)

#%% _____________________ Initialize model _____________________  

model_size = {'nb_annot_vec' : 16*16,          # Fixed = dimensions of VGG features map
              'embedding_len': 128,            # Dimension of the word embedding
              'lstm_len'     : 1024,           # Number of neurons of in hidden state of the LSTM
              'initSize'     : 32,              # Number of neurons in the NN that initialize hidden state
              'wordmap_len'  : len(word_map),  # Fixed = number of words in the word map
              }

decoder26 = decoder.Decoder(**model_size)


#%% _____________________ Training model _____________________

loss_fn = nn.CrossEntropyLoss()

args = {'device':device, 'decoder':decoder26}

trainer_parameters = {'learning_rate':1e-3,
                      'teacher_forcing_ratio':0.2,
                      'dropout': 0.05,
                     } 
trainer = Trainer(**args, **trainer_parameters)

#%%
trainer.trainNet(datasets['training'], data_loader, word_map, decoder26, loss_fn, 3000)

evaluate.computeBLEU(datasets['validation'], decoder26, word_map, BLEU_idx = 1)    # 39.83
evaluate.computeBLEU(datasets['training'], decoder26, word_map, BLEU_idx = 1)      # 46.51

trainer.trainNet(datasets['training'], data_loader, word_map, decoder26, loss_fn, 1000)

evaluate.computeBLEU(datasets['validation'], decoder24, word_map, BLEU_idx = 1)    # 38.48
evaluate.computeBLEU(datasets['training'], decoder24, word_map, BLEU_idx = 1)      # 42.00

trainer.trainNet(datasets['training'], data_loader, word_map, decoder22, loss_fn, 1000)


evaluate.computeBLEU(datasets['validation'], decoder22, word_map, BLEU_idx = 1)    # 44.31
evaluate.computeBLEU(datasets['validation'], decoder22, word_map, BLEU_idx = 2)    # 27.88
evaluate.computeBLEU(datasets['training'], decoder22, word_map, BLEU_idx = 1)      # 63.38


trainer.learning_rate  = 1e-4
trainer.trainNet(datasets['training'], data_loader, word_map, decoder22, loss_fn, 1000)

evaluate.computeBLEU(datasets['validation'], decoder22, word_map, BLEU_idx = 1)    # 44.57
evaluate.computeBLEU(datasets['validation'], decoder22, word_map, BLEU_idx = 2)    # 27.91
evaluate.computeBLEU(datasets['training'], decoder22, word_map, BLEU_idx = 1)      # 67.69


#%% _____________________ Evaluating model _____________________

evaluate.evaluateRandomly(datasets['training'], decoder22, word_map, 5, True)
evaluate.evaluateRandomly(datasets['test'], decoder23, word_map, 5, True)

BLEU_1_score = evaluate.computeBLEU(datasets['test'], decoder22, word_map, BLEU_idx = 1) # 46.00
BLEU_2_score = evaluate.computeBLEU(datasets['test'], decoder22, word_map, BLEU_idx = 2) # 28.62
BLEU_3_score = evaluate.computeBLEU(datasets['test'], decoder22, word_map, BLEU_idx = 3) # 18.49
BLEU_4_score = evaluate.computeBLEU(datasets['test'], decoder22, word_map, BLEU_idx = 4) # 12.09

#%% _____________________ Saving model _____________________

sl.save_all(23, decoder23, trainer)
 
plt.plot(trainer2.plot_losses)
plt.plot(trainer3.plot_losses)
plt.show()    
#%% _____________________ Loading model _____________________

model_size_load = {'nb_annot_vec':16*16,         # Fixed = dimensions of VGG features map
                  'embedding_len':128,           # Dimension of the word embedding
                  'lstm_len':1546,               # Number of neurons of in hidden state of the LSTM
                  'initSize':64,                 # Number of neurons in the NN that initialize hidden state of the LSTM
                  'wordmap_len':len(word_map)    # Fixed = number of words in the word map
                  }
decoder22 = decoder.Decoder(**model_size_load)
trainer2 = sl.load_all(22, decoder22)
#trainer.decoder_optimizer.load_state_dict(trainer2.decoder_optimizer.state_dict())

