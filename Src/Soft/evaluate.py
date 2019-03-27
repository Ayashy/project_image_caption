# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 15:24:05 2019

@author: perez
"""

import numpy as np
import torch
from torch.utils import data
import PIL
from PIL import ImageFilter
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import cv2 as cv
import skimage

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
        attn_weights = attn_weights.squeeze(0).cpu()
        
        prediction=''
        isEnded = False
        for line in scores[0]:
            idx=line.argmax().item()
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
                        _, sorted_idx = torch.sort(line)
                        idx = sorted_idx[1].item()
                        for key in word_map.keys():
                            if word_map[key]==idx:
                                next_word = str(key)
                                
                    prediction += ' ' + next_word
                
        caption = tens_to_word(caps, caplens, word_map)      
            
        return prediction, caption, attn_weights
    

''' Randomly select a few images, and compare predicted caption to references caption
        input :     _dataset        FlickrDataset (training, validation or test)
                    decoder         Decoder model
                    n               Number of random images to predict and compare
                    
        output:     print prediction and references.
'''
def evaluateRandomly(_dataset, decoder, word_map, n=10, visualize=False):
    caps_per_img = _dataset.caps_per_img
    decoder = decoder.to(device)
    img_indexes = np.random.randint(0, len(_dataset)//caps_per_img, n)
    
    for img_idx in img_indexes:
        print('\nImage n° ', str(img_idx), 'captions :')
        if visualize:
            ID = list(_dataset.captions.keys())[img_idx]
            original_img = PIL.Image.open('./raw_data/Flickr8k_data/'+ID+'.jpg')
            plt.title('original image')
            plt.axis('off')
            plt.imshow(original_img)
            
        for cap_idx in range(caps_per_img):
            idx = img_idx*caps_per_img + cap_idx
            img, caps, caplens = _dataset[idx]
            img, caps, caplens = img.unsqueeze(0), caps.unsqueeze(0), caplens.unsqueeze(0)
            caption = tens_to_word(caps, caplens, word_map)        
            print('>', caption)
            
        if visualize:
            image_to_tens = transforms.ToTensor()
            
            prediction, caption, attn_weights = predict(decoder, img, caps, caplens, word_map, False)
            words = prediction.split(' ')
            feature_map_size = int(np.sqrt(attn_weights.size(1)))  # Dimension of map : should be 16=sqrt(16x16)
            for i in range(2,len(words)):
                if (words[i]=='<end>'):
                    break
                
                plt.figure()                # To plot the attention weights on the original image
                plt.title(words[i])         # Word predicted for this attention weights
                plt.axis('off')

                original_img_tens = image_to_tens(original_img)                # Original image as a tensor
                H, W = original_img_tens.size(1), original_img_tens.size(2)  # Dimension of the original image
                
                attn_tens = attn_weights[i].view(1,feature_map_size,-1)
                attn_tens = torch.cat([attn_tens,attn_tens,attn_tens],dim=0) # 1 channel to RGB channel

                attn_img = np.array(attn_tens.permute(1,2,0))
                attn_img = skimage.transform.pyramid_expand(attn_img,upscale=16,sigma=20,multichannel=True)
                plt.imshow(np.array(original_img_tens.permute(1,2,0)))
                plt.imshow(100*cv.resize(attn_img,(W,H)),alpha=0.8)


        else:
            prediction, caption, _ = predict(decoder, img, caps, caplens, word_map, False)
        print('Predicted : ', prediction)
        plt.show()
        
    decoder = decoder.cpu()
     



def beam_search():
    return 0

    
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
        
    elif BLEU_idx == 2: # BLEU-2
        BLEU += corpus_bleu(refs, preds, weights=(.5,.5,0,0))      
        
    elif BLEU_idx == 3: # BLEU-3
        BLEU += corpus_bleu(refs, preds, weights=(.33,.33,.33,0))      
        
    elif BLEU_idx == 4: # BLEU-4
        BLEU += corpus_bleu(refs, preds, weights=(.25,.25,.25,.25))      
                
    #BLEU /= (len(img_subset) + 1e-19)     # Different from len(_dataset) !!
    
    if BLEU_idx == 0:
        bleu_str = ''
    else:
        bleu_str = '-' + str(BLEU_idx)
        
    print ('\nBLEU' + bleu_str + ' score : ', str(BLEU))
 
    return BLEU   
    
    
    
    
    
    
    
    
    