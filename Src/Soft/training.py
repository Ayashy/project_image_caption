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


from decoder import decode
import evaluate  # To show translation along training

teacher_forcing_ratio = 0.5

class Trainer:
    ''' Create trainer class :
        input :     device          torch device, either cpu or gpu (cuda)
                    learning rate   learning rate to train with
                    plot_losses     history of training loss
                    optimizer       decoder model optimizer, to resume training later
    '''
    def __init__(self, device, learning_rate, decoder, teacher_forcing_ratio=0.5, dropout=0.1):
        self.device = device
        self.learning_rate = learning_rate
        self.plot_losses = []
        self.decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.dropout = dropout
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
    def iter_training(self, imgs, caps, caplens, decoder, loss_fn, word_map):
        max_cap_len = 20  
        
        self.decoder_optimizer.zero_grad()
        
        teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        total_loss = 0.
    
        ''' Decoding entire sentence '''
        batch_size = imgs.size(0)
        
        SOS_token = word_map['<start>']
        output_word = decoder.initCaption(SOS_token, batch_size)
        decode_lengths = (max_cap_len + 1)*torch.ones(batch_size, dtype=torch.long, device=self.device)
        h, c = decoder.init_hidden_state(imgs,self.device)
      
        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(1,max_cap_len + 1):
            if teacher_forcing:
                output_word = caps[:,t-1].squeeze()            
            score, word, h, c, weights = decoder(imgs, output_word, h, c)

            target = caps[:,t]
            
            loss = loss_fn(score, target)
            total_loss += loss
            loss.backward(retain_graph=True)
        
        total_loss.backward()
        self.decoder_optimizer.step()
        
        self.decoder_optimizer.zero_grad()
        del imgs, caps, caplens, decode_lengths
        torch.cuda.empty_cache()
            
        return total_loss.item()

    ''' Performs model training
        input :     _dataset        FlickrDataset used for training
                    dataLoader      Data generator of the FlickrDataset for mini-batch training
                    decoder         decoder model
                    loss_fn         loss_function (e.g. Cross Entropy)
                    max_epoch       number of epoch to run (at fixed learning rate)
                    
        output :    show training loss
                    update decoder paramaters
    ''' 
    def trainNet(self,_dataset, dataLoader, word_map, decoder, loss_fn, max_epoch,
                 print_every=40, test_every=50000,plot_every=20):
                
        start = time.time()
        
        print_loss_total = 0    # Reset every print_every
        plot_loss_total = 0     # Reset every plot_every
              
        max_iter    = 10000           # Precaution
        epoch       = self.epoch
        start_epoch = self.epoch - 1
        
        for g in self.decoder_optimizer.param_groups:
            g['lr'] = self.learning_rate
            
        decoder.dropout_p = self.dropout
        decoder  = decoder.to(self.device)      # Pass model to device (GPU is available)
    
        for n_iter in range(max_iter):
            # Iterating through batches
            for i, (imgs, caps, caplens) in enumerate(dataLoader):
                caps = caps.to(device=self.device, dtype=torch.int64)               # Pass data to device (same as decoder)
                imgs, caplens = imgs.to(self.device), caplens.to(self.device)       # Pass data to device 
            
                loss = self.iter_training(imgs, caps, caplens, decoder, loss_fn, word_map)    # Update model parameters
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
        decoder.dropout_p = 0.
        
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
    
