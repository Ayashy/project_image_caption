import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from encoder import Encoder
from decoder import Decoder
from flickr_dataset import *
from utils import *


def train():
    """
    Training and validation.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")  

    # Model parameters
    data_folder = os.path.join('processed_data')
    embedding_len = 512  
    attention_len = 512  
    lstm_len = 512  
    caps_per_image=2
    batch_size = 32
    learning_rate = 4e-4  

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Decoder model
    decoder = Decoder(attention_len=attention_len,
                        embedding_len=embedding_len,
                        features_len=lstm_len,
                        wordmap_len=len(word_map))
    decoder = decoder.to(device)
    # We need an optimiser to update the model weights
    grad_params=filter(lambda p: p.requires_grad, decoder.parameters())
    decoder_optimizer = torch.optim.Adam(params=grad_params, lr=learning_rate)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Dataloaders are wrappers around datasets that help woth the learning.
    # Its not mandatory but its usefull so we might as well use it
    dataset=FlickrDataset(data_folder, 'TEST', caps_per_image)
    data_loader = torch.utils.data.DataLoader(
                FlickrDataset(data_folder, 'TEST', caps_per_image),
                batch_size=50, )

    for epoch in range(1,15):
        print( '----------------------------------- Epoch',epoch,'----------------------------------')
        for i,(imgs,caps,caplens) in enumerate(data_loader):
            caps = caps.to(device=device, dtype=torch.int64)               
            imgs, caplens = imgs.to(device), caplens.to(device)  
            #print( '----------------------------------- Batch',i,'----------------------------------')
            print(imgs.shape)
            scores, caps_sorted, decode_lengths, alphas, sort_ind=decoder.forward(imgs, caps, caplens)
            # Remove the <start> word
            targets = caps_sorted[:, 1:]

            prediction=''
            for line in scores[0]:
                idx=line.argmax()
                for key in word_map.keys():
                    if word_map[key]==idx:
                        prediction+=' '+str(key)
            caption=''
            for line in caps[sort_ind[0], 1:caplens[sort_ind[0]]+1]:
                for key in word_map.keys():
                    if word_map[key]==line:
                        caption+=' '+str(key)

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)
            #print('loss : ',loss.item())
            decoder_optimizer.zero_grad()

            loss.backward()
            decoder_optimizer.step()

            
        #print('--------------------------------------------------------------')
        print('Prediction : ',prediction)
        print('Truth : ',caption)
        #print('--------------------------------------------------------------')    



if __name__ == '__main__':
    
    #reduce_dataset(data_folder+'/train_images',data_folder+'/TRAIN_CAPTIONS.json',data_folder+'/DEV_CAPTIONS.json')

    train()
