# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 15:24:05 2019

@author: perez
"""

import numpy as np
import torch
from torch.utils import data

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu

from flickr_dataset import tens_to_word
from decoder import decode 

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


''' Generate a caption with the model from input image
        input :     imgs            input images (processed by VGG)         : Tensor(1,16,16,512)
                    caps            padded captions of the images (1/image) : Tensor(1, max_cap_size)
                    capslens        lengths of the captions for unpadding   : Liste(1)
                    
        output:     prediction      caption generated with model (string)
                    caption         reference caption of the image (string)
'''
def predict(decoder, img, caps, caplens, word_map, teacher_forcing=False): 
    caps = caps.to(device=device, dtype=torch.int64)
    img, caplens = img.to(device), caplens.to(device)
    
    with torch.no_grad():
        scores, caps, caplens, decode_lengths, attn_weights = decode(decoder, img, caps, 
                                                                     caplens, 20, word_map, teacher_forcing, False)

        prediction=''
        isEnded = False
        for line in scores[0]:
            idx=line.argmax()
            for key in word_map.keys():
                if word_map[key]==idx:
                    next_word = str(key)
                    if next_word == '<pad>':   # Do not print pad
                        next_word = ''
                    elif next_word == '<end>': # Do not prind <end> if more than one
                        if isEnded:
                            next_word = ''
                        else:
                            isEnded = True
                    elif next_word == '<unk>': # Look for the 2nd highest word
                        idx = torch.sort(line)[1]
                        for key in word_map.keys():
                            if word_map[key]==idx:
                                next_word = str(key)
                                
                    prediction += ' ' + next_word

        caption = tens_to_word(caps, caplens, word_map)      
            
        return prediction, caption
    

''' Randomly select a few images, and compare predicted caption to references caption
        input :     _dataset        FlickrDataset (training, validation or test)
                    decoder         Decoder model
                    n               Number of random images to predict and compare
                    
        output:     print prediction and references.
'''
def evaluateRandomly(_dataset, decoder, word_map, n=10, teacher_forcing=False):
    caps_per_img = _dataset.caps_per_img
    decoder = decoder.to(device)
    img_indexes = np.random.randint(0, len(_dataset)//caps_per_img, n)
    
    for img_idx in img_indexes:
        print('\nImage n° ', str(img_idx), 'captions :')
        for cap_idx in range(caps_per_img):
            idx = img_idx*caps_per_img + cap_idx
            img, caps, caplens = _dataset[idx]
            img, caps, caplens = img.unsqueeze(0), caps.unsqueeze(0), caplens.unsqueeze(0)
            caption = tens_to_word(caps, caplens, word_map)        
            print('>', caption)
            
        prediction, caption = predict(decoder, img, caps, caplens, word_map, teacher_forcing)
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
    
    limitation_words = [word_map['<start>'], word_map['<end>'], word_map['<pad>']]
    decoder = decoder.to(device)
    refs = list()
    preds = list()
    for i, (imgs, caps, caplens) in enumerate(data_loader):           
        caps = caps.to(device=device, dtype=torch.int64)
        imgs, caplens = imgs.to(device), caplens.to(device)
        
        # Calculating the model scores of every img in the dataset in one batch
        # to further accelerate computation on GPU
        with torch.no_grad():
            scores, caps, caplens, decode_lengths, attn_weights = decode(decoder, imgs, caps, caplens, 20, word_map, False, False)
            pred = scores.argmax(dim=2)
            preds += [[x.item() for x in pred[j] if x not in limitation_words] for j in range(scores.size(0))]
            
            
        for img_idx in range(i*batch_sz,(i+1)*batch_sz):
            if img_idx > 1 and img_idx % 100 == 0:
                print('evaluating ', str(img_idx), 'th image')
                    
            # Output every caption of the image as str    
            ref = list()
            for cap_idx in range(caps_per_img):
                idx = img_idx*caps_per_img + cap_idx
                img, caps, caplens = _dataset[idx]
                this_ref = [x.item() for x in caps if x not in limitation_words]
                ref.append(this_ref)
                
            refs.append(ref)
            
    # Compute BLEU score of the image from ref. captions (ref) and predictions (pred)   
    if BLEU_idx == 0:   # Cumulated 4-gram BLEU score
        BLEU += corpus_bleu(refs, preds)      # Score = average of every sentence
        
    elif BLEU_idx == 1: # BLEU-1
        BLEU += corpus_bleu(refs, preds, weights=(1,0,0,0))      
        
    elif BLEU_idx == 1: # BLEU-2
        BLEU += corpus_bleu(refs, preds, weights=(.5,.5,0,0))      
        
    elif BLEU_idx == 1: # BLEU-3
        BLEU += corpus_bleu(refs, preds, weights=(.33,.33,.33,0))      
        
    elif BLEU_idx == 1: # BLEU-4
        BLEU += corpus_bleu(refs, preds, weights=(.25,.25,.25,.25))      
                
    #BLEU /= (len(img_subset) + 1e-19)     # Different from len(_dataset) !!
    
    if BLEU_idx == 0:
        bleu_str = ''
    else:
        bleu_str = '-' + str(BLEU_idx)
        
    print ('\nBLEU' + bleu_str + ' score : ', str(BLEU))
 
    return refs, preds, BLEU   
    
    
    
    
    
    
    
    
    