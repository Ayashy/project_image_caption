import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class FlickrDataset(Dataset):
    """
    Helper class to load up the data.
    """

    def __init__(self,data_folder,split,caps_per_img=2):

        self.split = split
        self.data_folder=data_folder
        self.caps_per_img=caps_per_img

        # Load captions
        #with open(os.path.join(data_folder, self.split + '_CAPTIONS.json'), 'r') as j:
        with open(os.path.join(data_folder , self.split+'_CAPTIONS.json'), 'r') as j:
            self.captions = json.load(j)

        # Length of the dataset
        self.dataset_size = len(self.captions)*caps_per_img


    def __getitem__(self, i):
        
        if self.split=='DEV':
            self.split='TRAIN'
            
        ID=list(self.captions.keys())[i//self.caps_per_img]
        img_path=os.path.join(self.data_folder , self.split+'_images',ID+'.pt')
        img=torch.load(img_path)
        
        caption = torch.Tensor(self.captions[ID]["caps"][i%2]).to(dtype=torch.int16)
        caplen = torch.Tensor([self.captions[ID]["caplens"][i%2]]).to(dtype=torch.int16)

        if self.split!='TRAIN':
            allcaps=torch.Tensor(self.captions[ID]["caps"]).to(dtype=torch.int16)
            return img, caption,caplen,allcaps

        return img, caption,caplen

    def __len__(self):
        return self.dataset_size



class CocoDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split):

        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']
        self.caps_per_img = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)


        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])-2

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen

    def __len__(self):
        return self.dataset_size


if __name__ == "__main__":
    dataset=CocoDataset(os.path.join('processed_data','Coco'),'TRAIN')
    print(dataset[0])
    print('something')
