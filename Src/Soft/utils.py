from collections import Counter
import os
import json
import h5py
import string
from skimage.transform import resize
import skimage.io as io
import numpy as np
from encoder import Encoder
import torch
from tqdm import tqdm
from random import seed, choice, sample
from scipy.misc import imread, imresize


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
    max_cap_len = 20
    output_folder = './processed_data'
    caps_per_img = 2

    # Loading split IDs
    train_ids = load_doc('./raw_data/Flickr8k_text/Flickr8k.trainImages.txt')
    train_ids = [x.split('.')[0] for x in train_ids.split('\n')]
    eval_ids = load_doc('./raw_data/Flickr8k_text/Flickr8k.devImages.txt')
    eval_ids = [x.split('.')[0] for x in eval_ids.split('\n')]
    test_ids = load_doc('./raw_data/Flickr8k_text/Flickr8k.testImages.txt')
    test_ids = [x.split('.')[0] for x in test_ids.split('\n')]

    # Generating proccessed images then storing them
    idx = 1
    for ID in train_ids:
        if ID != '':
            image = proccess_image('./raw_data/Flickr8k_data/'+ID+'.jpg', enc)

            image = image.cpu()
            torch.save(image, './processed_data/TRAIN_images/'+ID+'.pt')

            if idx % 100 == 0:
                print('Generated ', str(idx), 'th training image')
            idx += 1

    idx = 1
    for ID in eval_ids:
        if ID != '':
            image = proccess_image('./raw_data/Flickr8k_data/'+ID+'.jpg', enc)
            image = image.cpu()
            torch.save(image, './processed_data/EVAL_images/'+ID+'.pt')

            if idx % 100 == 0:
                print('Generated ', str(idx), 'th validation image')
            idx += 1

    idx = 1
    for ID in test_ids:
        if ID != '':
            image = proccess_image('./raw_data/Flickr8k_data/'+ID+'.jpg', enc)
            image = image.cpu()
            torch.save(image, './processed_data/TEST_images/'+ID+'.pt')

            if idx % 100 == 0:
                print('Generated ', str(idx), 'th test image')
            idx += 1

    # Loading captions
    data = load_doc('./raw_data/Flickr8k_text/Flickr8k.token.txt')
    train_captions, eval_captions, test_captions = load_captions(
        data, train_ids, eval_ids, test_ids, caps_per_img)

    # Generating the wordmap then saving it to file
    wordmap = generate_wordmap(
        [train_captions, eval_captions, test_captions], min_frequency)
    with open(os.path.join(output_folder, 'WORDMAP.json'), 'w') as j:
        json.dump(wordmap, j)

    # Process captions then store in file
    train_captions, eval_captions, test_captions = process_captions(
        [train_captions, eval_captions, test_captions], wordmap, max_cap_len)
    with open(os.path.join(output_folder, 'TRAIN_CAPTIONS.json'), 'w') as j:
        json.dump(train_captions, j)
    with open(os.path.join(output_folder, 'EVAL_CAPTIONS.json'), 'w') as j:
        json.dump(eval_captions, j)
    with open(os.path.join(output_folder, 'TEST_CAPTIONS.json'), 'w') as j:
        json.dump(test_captions, j)


def generate_wordmap(splits, min_frequency):
    word_counter = Counter()
    for split in splits:
        for line in split.values():
            for cap in line:
                word_counter.update(cap.split(' '))
    words = [x for x in word_counter.keys() if word_counter[x] > min_frequency]
    wordmap = {k: v + 1 for v, k in enumerate(words)}
    wordmap['<unk>'] = len(wordmap) + 1
    wordmap['<start>'] = len(wordmap) + 1
    wordmap['<end>'] = len(wordmap) + 1
    wordmap['<pad>'] = 0
    return wordmap


def process_captions(splits, wordmap, max_cap_len):
    for split in splits:
        for image in split.keys():
            captions = split[image]
            new_captions = []
            caplens = []
            for i, cap in enumerate(captions):
                cap = cap.split()

                if len(cap) > max_cap_len:
                    caplens.append(max_cap_len)
                    cap = [wordmap['<start>']] + [wordmap.get(
                        word, wordmap['<unk>']) for word in cap[:max_cap_len]] + [wordmap['<end>']]
                else:
                    caplens.append(len(cap))
                    cap = [wordmap['<start>']] + [wordmap.get(word, wordmap['<unk>']) for word in cap] + [
                        wordmap['<end>']] + [wordmap['<pad>']] * (max_cap_len - len(cap))

                new_captions.append(cap)
            split[image] = {'caps': new_captions, 'caplens': caplens}
    return splits

# Should we remove the a and 's?


def load_captions(data, train_ids, eval_ids, test_ids, caps_per_img):
    table = str.maketrans('', '', string.punctuation)
    train_captions = {}
    eval_captions = {}
    test_captions = {}
    for line in data.split('\n'):
        tokens = line.split()
        image_id, image_cap = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_cap = [w.translate(table) for w in image_cap]
        #image_cap=[w for w in image_cap if len(w)>1]
        image_cap = ' '.join(image_cap).lower()
        if image_id in train_ids:
            if image_id not in train_captions.keys():
                train_captions[image_id] = []
            if (len(train_captions[image_id]) <= caps_per_img):
                train_captions[image_id].append(image_cap)

        if image_id in eval_ids:
            if image_id not in eval_captions.keys():
                eval_captions[image_id] = []
            if (len(eval_captions[image_id]) <= caps_per_img):
                eval_captions[image_id].append(image_cap)

        if image_id in test_ids:
            if image_id not in test_captions.keys():
                test_captions[image_id] = []
            if (len(test_captions[image_id]) <= caps_per_img):
                test_captions[image_id].append(image_cap)

    return train_captions, eval_captions, test_captions


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
    img = resize(img, (224, 224))
    img = img.transpose(2, 0, 1)
    img = torch.Tensor(img)
    img = img.reshape(1, 3, img.shape[1], img.shape[2])
    img = img.to(device)
    return enc(img)


# This func is only used for devoloppement purposes
def reduce_dataset(image_path, caption_path, output_path):
    from os import walk
    f = []
    for (dirpath, dirnames, filenames) in walk(image_path):
        f.extend(filenames)
        break
    with open(os.path.join(caption_path), 'r') as j:
        captions = json.load(j)

    new_captions = {}
    for i in f:
        new_captions[i[:-3]] = captions[i[:-3]]

    with open(os.path.join(output_path), 'w') as j:
        json.dump(new_captions, j)


def preprocess_coco_data():

    # Parameters
    image_folder= os.path.join('raw_data','Coco_data')
    captions_per_image=3
    min_word_freq=2
    output_folder=os.path.join('processed_data','Coco')
    max_len=100

    # Load VGG
    enc = Encoder().to(device)

    with open( os.path.join('raw_data','Coco_text','dataset_coco.json'), 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])
        
        if len(captions) == 0:
            continue
        path = os.path.join(image_folder, img['filepath'], img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP.json'), 'w') as j:
        json.dump(word_map, j)

    
    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:
        print('Processing ',split,'data : ------------------------------')
        if os.path.exists(os.path.join(output_folder, split + '_IMAGES_' + '.hdf5')):
                os.remove(os.path.join(output_folder, split + '_IMAGES_' + '.hdf5'))
        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset(
                'images', (len(impaths), 14, 14, 512), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i])
                                            for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Save image to HDF5 file
                images[i] =proccess_image(impaths[i], enc).cpu().detach()
                print('Processed',split,'image number'+str(i))


                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * \
                captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' +  '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' +  '.json'), 'w') as j:
                json.dump(caplens, j)


if __name__ == "__main__":
    #preprocess_flickr_data()
    preprocess_coco_data()