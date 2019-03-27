import torch
from torch.utils.data import Dataset
import json


class FlickrDataset(Dataset):
    """
    Helper class to load up the data.
    """

    def __init__(self,data_folder,split,caps_per_img,max_cap_len=20):

        self.split = split
        self.data_folder    = data_folder
        self.caps_per_img   = caps_per_img
        self.max_cap_len    = max_cap_len

        # Load captions
        #with open(os.path.join(data_folder, self.split + '_CAPTIONS.json'), 'r') as j:
        with open(data_folder + '\\' + self.split + '_CAPTIONS.json', 'r') as j:
            self.captions = json.load(j)

        # Length of the dataset
        self.dataset_size = len(self.captions)*caps_per_img


    def __getitem__(self, i):

        if self.split=='DEV':
            self.split='TRAIN'
            
        ID=list(self.captions.keys())[i//self.caps_per_img]
        img_path=self.data_folder+'\\'+self.split+'_images\\'+ID+'.pt'
        img=torch.load(img_path)
        
        caption = torch.Tensor(self.captions[ID]["caps"][i%2]).to(dtype=torch.int16)
        caplen = torch.Tensor([self.captions[ID]["caplens"][i%2]]).to(dtype=torch.int16)

        return img, caption,caplen

    def __len__(self):
        return self.dataset_size


def tens_to_word(caps, caplens, word_map):
    caption=''

    for line in caps[0, 1:caplens[0]+1]:
        for key in word_map.keys():
            if word_map[key]==line:
                caption += ' ' + str(key)

    return caption
