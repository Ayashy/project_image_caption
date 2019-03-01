# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 15:24:05 2019

@author: perez
"""

import numpy as np
import torch
from torch.utils import data

from nltk.translate.bleu_score import sentence_bleu

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


''' Generate a caption with the model from input image
        input :     imgs            input images (processed by VGG)         : Tensor(1,16,16,512)
                    caps            padded captions of the images (1/image) : Tensor(1, max_cap_size)
                    capslens        lengths of the captions for unpadding   : Liste(1)
                    
        output:     prediction      caption generated with model (string)
                    caption         reference caption of the image (string)
'''
def predict(decoder, img, caps, caplens, word_map): 
    caps = caps.to(device=device, dtype=torch.int64)
    img, caplens = img.to(device), caplens.to(device)
    
    with torch.no_grad():
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(img, caps, caplens)

        prediction=''
        for line in scores[0]:
            idx=line.argmax()
            for key in word_map.keys():
                if word_map[key]==idx:
                    prediction += ' ' + str(key)

        caption = print_caption(caps, caplens, word_map)      
        '''
        caption=''
        for line in caps[sort_ind[0], 1:caplens[sort_ind[0]]+1]:
            for key in word_map.keys():
                if word_map[key]==line:
                    caption += ' ' + str(key)
        '''
            
        return prediction, caption

def print_caption(caps, caplens, word_map):
    caption=''
    for line in caps[0, 1:caplens[0]+1]:
        for key in word_map.keys():
            if word_map[key]==line:
                caption += ' ' + str(key)

    return caption

''' Randomly select a few images, and compare predicted caption to references caption
        input :     _dataset        FlickrDataset (training, validation or test)
                    decoder         Decoder model
                    n               Number of random images to predict and compare
                    
        output:     print prediction and references
'''
def evaluateRandomly(_dataset, decoder, word_map, n=10):
    caps_per_img = _dataset.caps_per_img
    decoder = decoder.to(device)
    img_indexes = np.random.randint(0, len(_dataset)//caps_per_img, n)
    
    for img_idx in img_indexes:
        print('\nImage n° ', str(img_idx), 'captions :')
        for cap_idx in range(caps_per_img):
            idx = img_idx*caps_per_img + cap_idx
            img, caps, caplens = _dataset[idx]
            img, caps, caplens = img.unsqueeze(0), caps.unsqueeze(0), caplens.unsqueeze(0)
            caption = print_caption(caps, caplens, word_map)        
            print('>', caption)
            
        prediction, caption = predict(decoder, img, caps, caplens, word_map)
        print('Predicted : ', prediction)
        
    decoder = decoder.cpu()
     

def beam_search():
    return 0

     
''' Compute BLEU score (without beam search)
        input :     _dataset        FlickrDataset (training, validation or test)
                    decoder         Decoder model
                    BLEU_idx        idx of the BLEU score : BLEU-1, BLEU-2, BLEU-3, BLEU-4
                                    default is cumulative BLEU-4 score
                    
        output:     BLEU            BLEU score on the dataset
'''   
'''
def computeBLEU(_dataset, decoder, word_map, BLEU_idx = 0):
    BLEU = 0.
    
    caps_per_img = _dataset.caps_per_img
    decoder = decoder.to(device)
    
    for img_idx in range(len(_dataset)//caps_per_img):
        if img_idx % 100 == 0:
            print('evaluating ', str(img_idx), 'th image')
        ref = [[]]
        for cap_idx in range(caps_per_img):
            idx = img_idx*caps_per_img + cap_idx
            img, caps, caplens = _dataset[idx]
            
            img, caps, caplens = img.unsqueeze(0), caps.unsqueeze(0), caplens.unsqueeze(0)
            caption = print_caption(caps, caplens, word_map)            
            ref.append(caption.split(' '))
            
        prediction, caption = predict(decoder, img, caps, caplens, word_map)
        pred = prediction.split(' ')
        
        if BLEU_idx == 0:   # Cumulated 4-gram BLEU score
            BLEU += sentence_bleu(ref, pred)      # Score = average of every sentence
            
        elif BLEU_idx == 1: # BLEU-1
            BLEU += sentence_bleu(ref, pred, weights=(1,0,0,0))      
            
        elif BLEU_idx == 1: # BLEU-2
            BLEU += sentence_bleu(ref, pred, weights=(0,1,0,0))      
            
        elif BLEU_idx == 1: # BLEU-3
            BLEU += sentence_bleu(ref, pred, weights=(0,0,1,0))      
            
        elif BLEU_idx == 1: # BLEU-4
            BLEU += sentence_bleu(ref, pred, weights=(0,0,0,1))      
            
    BLEU /= (len(_dataset) + 1e-19)
    
    if BLEU_idx == 0:
        bleu_str = ''
    else:
        bleu_str = '-' + str(BLEU_idx)
        
    print ('\nBLEU' + bleu_str + ' score : ', str(BLEU))
 
    return BLEU
'''
    
def computeBLEU(_dataset, decoder, word_map, BLEU_idx = 0):
    print('\nComputing BLEU score ...')
    BLEU = 0.
    
    caps_per_img = _dataset.caps_per_img
    N = len(_dataset)
    
    img_idxs = [i*caps_per_img for i in range(N//caps_per_img)]
    img_subset = data.Subset(_dataset,img_idxs)                         # Dataset with only one caption per image
    batch_sz = 100
    data_loader = data.DataLoader(img_subset, batch_size = batch_sz)     # Limit batch size for CUDA memory
    
    decoder = decoder.to(device)
    for i, (imgs, caps, caplens) in enumerate(data_loader):           
        caps = caps.to(device=device, dtype=torch.int64)
        imgs, caplens = imgs.to(device), caplens.to(device)
        
        # Calculating the model scores of every img in the dataset in one batch
        # to further accelerate computation on GPU
        with torch.no_grad():
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
    
        for img_idx in range(i*batch_sz,(i+1)*batch_sz):
            if img_idx > 1 and img_idx % 100 == 0:
                print('evaluating ', str(img_idx), 'th image')
                    
            # Output every caption of the image as str    
            ref = [[]]
            for cap_idx in range(caps_per_img):
                idx = img_idx*caps_per_img + cap_idx
                img, caps, caplens = _dataset[idx]
                
                img, caps, caplens = img.unsqueeze(0), caps.unsqueeze(0), caplens.unsqueeze(0)
                caption = print_caption(caps, caplens, word_map)            
                ref.append(caption.split(' '))
            
            # Predict caption with model as str (from previously calculated scores)
            prediction=''
            for line in scores[img_idx - i*batch_sz]:
                idx=line.argmax()
                for key in word_map.keys():
                    if word_map[key]==idx:
                        prediction += ' ' + str(key)
            pred = prediction.split(' ')
            
            # Compute BLEU score of the image from ref. captions (ref) and predictions (pred)   
            if BLEU_idx == 0:   # Cumulated 4-gram BLEU score
                BLEU += sentence_bleu(ref, pred)      # Score = average of every sentence
                
            elif BLEU_idx == 1: # BLEU-1
                BLEU += sentence_bleu(ref, pred, weights=(1,0,0,0))      
                
            elif BLEU_idx == 1: # BLEU-2
                BLEU += sentence_bleu(ref, pred, weights=(0,1,0,0))      
                
            elif BLEU_idx == 1: # BLEU-3
                BLEU += sentence_bleu(ref, pred, weights=(0,0,1,0))      
                
            elif BLEU_idx == 1: # BLEU-4
                BLEU += sentence_bleu(ref, pred, weights=(0,0,0,1))      
                
        BLEU /= (len(_dataset) + 1e-19)
        
        if BLEU_idx == 0:
            bleu_str = ''
        else:
            bleu_str = '-' + str(BLEU_idx)
            
        print ('\nBLEU' + bleu_str + ' score : ', str(BLEU))
 
    return BLEU   
    
    
    
    
    
    
    
    
    