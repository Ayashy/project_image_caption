from collections import Counter
import os
import json
import string
from skimage.transform import resize
import skimage.io as io
import numpy as np
from encoder import Encoder
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def preprocess_flickr_data():
    """
    This function is used to generate input files from raw data files. It generates 3 types of files:

    Images : we can either load them raw, vector h5py or generate the features using the encoder
    Captions : We sample a number of captions per image then store them as Json
    Caption lenghts : the lenghts of each captions, usefull to know when to stop the process ?!

    The data is also split into train evaluate and test
    """
    # Load VGG
    enc = Encoder().to(device)

    min_frequency = 2
    max_cap_len=20
    output_folder='./processed_data'
    caps_per_img=2

    # Loading split IDs
    train_ids=load_doc('./raw_data/Flickr8k_text/Flickr8k.trainImages.txt')
    train_ids=[x.split('.')[0] for x in train_ids.split('\n')]
    eval_ids=load_doc('./raw_data/Flickr8k_text/Flickr8k.devImages.txt')
    eval_ids=[x.split('.')[0] for x in eval_ids.split('\n')]
    test_ids=load_doc('./raw_data/Flickr8k_text/Flickr8k.testImages.txt')
    test_ids=[x.split('.')[0] for x in test_ids.split('\n')]

    # Generating proccessed images then storing them
    idx = 1
    for ID in train_ids:
        if ID != '':
            image=proccess_image('./raw_data/Flickr8k_data/'+ID+'.jpg', enc)

            image = image.cpu()
            torch.save(image, './processed_data/train_images/'+ID+'.pt')
            
            if idx % 100 == 0:
                print('Generated ', str(idx), 'th training image')
            idx += 1
            
    idx = 1
    for ID in eval_ids:
        if ID != '':
            image=proccess_image('./raw_data/Flickr8k_data/'+ID+'.jpg', enc)
            image = image.cpu()
            torch.save(image, './processed_data/eval_images/'+ID+'.pt')
            
            if idx % 100 == 0:
                print('Generated ', str(idx), 'th validation image')
            idx += 1
            
    idx = 1       
    for ID in test_ids:
        if ID != '':
            image=proccess_image('./raw_data/Flickr8k_data/'+ID+'.jpg', enc)
            image = image.cpu()
            torch.save(image, './processed_data/test_images/'+ID+'.pt')
            
            if idx % 100 == 0:
                print('Generated ', str(idx), 'th test image')
            idx += 1

    # Loading captions
    data=load_doc('./raw_data/Flickr8k_text/Flickr8k.token.txt')
    train_captions,eval_captions,test_captions = load_captions(data,train_ids,eval_ids,test_ids,caps_per_img)
    
    # Generating the wordmap then saving it to file
    wordmap=generate_wordmap([train_captions,eval_captions,test_captions],min_frequency)
    with open(os.path.join(output_folder, 'WORDMAP.json'), 'w') as j:
        json.dump(wordmap, j)

    # Process captions then store in file
    train_captions,eval_captions,test_captions=process_captions([train_captions,eval_captions,test_captions],wordmap,max_cap_len)
    with open(os.path.join(output_folder, 'TRAIN_CAPTIONS.json'), 'w') as j:
        json.dump(train_captions, j)
    with open(os.path.join(output_folder, 'EVAL_CAPTIONS.json'), 'w') as j:
        json.dump(eval_captions, j)
    with open(os.path.join(output_folder, 'TEST_CAPTIONS.json'), 'w') as j:
        json.dump(test_captions, j)

def generate_wordmap(splits,min_frequency):
    word_counter=Counter()
    for split in splits:
        for line in split.values():
            for cap in line:
                word_counter.update(cap.split(' '))
    words= [ x for x  in word_counter.keys() if word_counter[x]>min_frequency]
    wordmap = {k: v + 1 for v, k in enumerate(words)}
    wordmap['<unk>'] = len(wordmap) + 1
    wordmap['<start>'] = len(wordmap) + 1
    wordmap['<end>'] = len(wordmap) + 1
    wordmap['<pad>'] = 0    
    return wordmap
    
def process_captions(splits,wordmap,max_cap_len):
    for split in splits:
        for image in split.keys():
            captions=split[image]
            new_captions=[]
            caplens=[]
            for i,cap in enumerate(captions):
                cap=cap.split()
                
                if len(cap) > max_cap_len:
                    caplens.append(max_cap_len)
                    cap = [wordmap['<start>']] + [wordmap.get(word, wordmap['<unk>']) for word in cap[:max_cap_len]] + [wordmap['<end>']]
                else:
                    caplens.append(len(cap))
                    cap=[wordmap['<start>']] + [wordmap.get(word, wordmap['<unk>']) for word in cap] + [
                            wordmap['<end>']] + [wordmap['<pad>']] * (max_cap_len - len(cap))
                    
                new_captions.append(cap)
            split[image]={'caps':new_captions,'caplens':caplens}
    return splits

# Should we remove the a and 's?
def load_captions(data,train_ids,eval_ids,test_ids,caps_per_img):
    table = str.maketrans('', '', string.punctuation)
    train_captions = {}
    eval_captions = {}
    test_captions = {}
    for line in data.split('\n'):
        tokens = line.split()
        image_id, image_cap = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_cap=[w.translate(table) for w in image_cap]
        #image_cap=[w for w in image_cap if len(w)>1]
        image_cap = ' '.join(image_cap).lower()
        if image_id in train_ids:
            if image_id not in train_captions.keys():
                train_captions[image_id] = []
            if (len(train_captions[image_id])<=caps_per_img):
                train_captions[image_id].append(image_cap)

        if image_id in eval_ids:
            if image_id not in eval_captions.keys():
                eval_captions[image_id] = []
            if (len(eval_captions[image_id])<=caps_per_img):
                eval_captions[image_id].append(image_cap)

        if image_id in test_ids:
            if image_id not in test_captions.keys():
                test_captions[image_id] = []
            if (len(test_captions[image_id])<=caps_per_img):
                test_captions[image_id].append(image_cap)

    return train_captions,eval_captions,test_captions

def load_doc(filename):
	"""
    Helper function to load a file as a string
    """
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

def proccess_image(path, enc):
    img = io.imread(path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = resize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = torch.Tensor(img)
    img = img.reshape(1,3,img.shape[1],img.shape[2])
    img = img.to(device)
    return enc(img).cpu()


# This func is only used for devoloppement purposes
def reduce_dataset(image_path,caption_path,output_path):
    from os import walk
    f = []
    for (dirpath, dirnames, filenames) in walk(image_path):
        f.extend(filenames)
        break
    with open(os.path.join(caption_path), 'r') as j:
            captions = json.load(j)
    
    new_captions={}
    for i in f:
        new_captions[i[:-3]]=captions[i[:-3]]
    
    with open(os.path.join(output_path), 'w') as j:
        json.dump(new_captions, j)

if __name__ == "__main__":
    preprocess_flickr_data()
