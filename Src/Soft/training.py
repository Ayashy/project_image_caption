# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 15:14:55 2019

@author: perez
"""

import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence


import evaluate  # To show translation along training

SOS_token = 0
EOS_token = 1

teacher_forcing_ratio = 0.5

class Trainer:
    ''' Create trainer class :
        input :     device          torch device, either cpu or gpu (cuda)
                    learning rate   learning rate to train with
                    plot_losses     history of training loss
                    optimizer       decoder model optimizer, to resume training later
    '''
    def __init__(self, device, learning_rate, decoder):
        self.device = device
        self.learning_rate = learning_rate
        self.plot_losses = []
        self.decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
        self.epoch = 1

    ''' Performs an update of the model (decoder) parameters for a given input batch
        input :     imgs            input images (processed by VGG)         : Tensor(batch_size,16,16,512)
                    caps            padded captions of the images (1/image) : Tensor(batch_size, max_cap_size)
                    capslens        lengths of the captions for unpadding   : Liste(batch_size)
                    decoder         decoder model
                    loss_fn         loss_function (e.g. Cross Entropy)
                    
        output:     loss            loss value for the given batch
                    update decoder paramaters
    '''    
    def iter_training(self, imgs, caps, caplens, decoder, loss_fn):
           
        decoder_optimizer = self.decoder_optimizer 
        decoder_optimizer.zero_grad()
        
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        loss = 0.
    
        ''' Without teacher forcing '''
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
        targets = caps_sorted[:, 1:]
        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this        
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        
        loss = loss_fn(scores, targets)
            
        '''
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, 
                                                                            encoder_outputs, batch_sz)
                loss += loss_fn(decoder_output, target_tensor[di,:])
                decoder_input = target_tensor[di,:]  # Teacher forcing
    
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, 
                                                                            encoder_outputs, batch_sz)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
    
                loss += loss_fn(decoder_output, target_tensor[di,:])
                #if decoder_input.item() == EOS_token:
                #    break
        '''
        loss.backward()
    
        decoder_optimizer.step()
        
        self.decoder_optimizer = decoder_optimizer
    
        return loss.item()

    ''' Performs model training
        input :     _dataset        FlickrDataset used for training
                    dataLoader      Data generator of the FlickrDataset for mini-batch training
                    decoder         decoder model
                    loss_fn         loss_function (e.g. Cross Entropy)
                    max_epoch       number of epoch to run (at fixed learning rate)
                    
        output :    show training loss
                    update decoder paramaters
    ''' 
    def trainNet(self,_dataset, dataLoader, decoder, loss_fn, max_epoch,
                 print_every=10, test_every=50000,plot_every=10):
                
        start = time.time()
        
        print_loss_total = 0    # Reset every print_every
        plot_loss_total = 0     # Reset every plot_every
              
        max_iter    = 10000           # Precaution
        epoch       = self.epoch
        start_epoch = self.epoch - 1
    
        decoder  = decoder.to(self.device)      # Pass model to device (GPU is available)
    
        for n_iter in range(max_iter):
            # Iterating through batches
            for i, (imgs, caps, caplens) in enumerate(dataLoader):
                caps = caps.to(device=self.device, dtype=torch.int64)               # Pass data to device (same as decoder)
                imgs, caplens = imgs.to(self.device), caplens.to(self.device)       # Pass data to device 
            
                loss = self.iter_training(imgs, caps, caplens, decoder, loss_fn)    # Update model parameters
                print_loss_total += loss
                plot_loss_total += loss
        
                if epoch % print_every == 0:   # Print advancement of the training (time spent and remaining)
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print('%s (%d %d%%) %.4f' % (timeSince(start, (epoch-start_epoch) / max_epoch),
                                                 (epoch-start_epoch), (epoch-start_epoch) / max_epoch * 100, print_loss_avg))
                    
                if epoch % test_every == 0:
                    print('\nRandom translations :')
                    evaluate.evaluateRandomly(_dataset, decoder, n=5)
        
                if epoch % plot_every == 0:    # Loss curve is smoothed : averaged on <print_every> iterations
                    plot_loss_avg = plot_loss_total / plot_every
                    self.plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0
                    
                epoch += 1
                
                del imgs, caps, caplens
                
                if (epoch - start_epoch >= max_epoch):
                    break 
                
            if (epoch - start_epoch >= max_epoch):
                    break
    
        decoder = decoder.cpu()
    
        self.epoch = epoch
        showPlot(self.plot_losses)      # Print training loss history



''' plot functions to see learning advancement over time '''
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
    
    
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    
