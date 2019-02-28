# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 16:11:23 2019

@author: perez
"""

import numpy as np
import torch
from torch import optim

from training import Trainer

def save_decoder(idx, decoder):
    return 0

def save_all(idx, decoder, trainer):
    # Saving torch decoder and optimizer
    dec_str = 'decoder'
    opt_str = 'optimizer'
    path = './models/soft' + str(idx) + '.pt'
    torch.save({dec_str: decoder.state_dict(), 
                opt_str: (trainer.decoder_optimizer).state_dict(),},
                path)
    
    # Saving current epoch and learning rate
    epoch_lr_path = './models/epoch_lr' + str(idx) + '.out'
    epoch = np.array([trainer.epoch, trainer.learning_rate])
    np.savetxt(epoch_lr_path, epoch, delimiter=',')
    
    # Saving plot loss history
    plot_path = './models/loss' + str(idx) + '.out'
    plot_losses = np.array(trainer.plot_losses)
    np.savetxt(plot_path, plot_losses, delimiter=',')
    
    print ('Succesfully saved model and trainer')
    
    
def load_all(idx, decoder):
    # Load epoch and learning rate
    epoch_lr = np.loadtxt('./models/epoch_lr' + str(idx) + '.out')
    epoch, learning_rate = epoch_lr[0], epoch_lr[1]
    
    # Loading training loss history
    plot_losses = np.loadtxt('./models/loss' + str(idx) + '.out')
    plot_losses = [loss for loss in plot_losses]
    
    # Load torch decoder and optimizer
    torch_path = './models/soft' + str(idx) + '.pt'
    checkpt = torch.load(torch_path)
    decoder.load_state_dict(checkpt['decoder'])
    
    optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpt['optimizer'])
        
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    trainer = Trainer(device, learning_rate, decoder)
    trainer.decoder_optimizer = optimizer
    trainer.plot_losses = plot_losses
    trainer.epoch = epoch
    
    return trainer