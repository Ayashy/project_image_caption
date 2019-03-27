import torch
from torch import nn
from torch.autograd import Variable
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

from attention import Attention
from flickr_dataset import tens_to_word

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class Decoder(nn.Module):
    """
    Implements the decoder model. 
    it's based on an lstm and uses attention.
    """

    def __init__(self, embedding_len, lstm_len, wordmap_len, initSize, nb_annot_vec=196, annot_vec_len=512, dropout_p=0.1):
        """
        Params:
            - attention_len: size of attention network
            - embedding_len: size of the embedding vector
            - features_len: size of the images features (14,14,512)
            - wordmap_len: number of words in the vocabulary
            - lstm_len: size of the lstm network
        """
        super(Decoder, self).__init__()

        self.lstm_len = lstm_len
        self.embedding_len  = embedding_len
        self.nb_annot_vec   = nb_annot_vec
        self.annot_vec_len  = annot_vec_len
        self.wordmap_len    = wordmap_len
        self.initSize = initSize
        self.dropout_p = dropout_p
        
        # Hidden and context vector initialization MLP (multi-layers perceptrons)
        self.init_h_MLP = nn.Sequential(nn.Linear(annot_vec_len, initSize), nn.Linear(initSize,lstm_len))
        self.init_c_MLP = nn.Sequential(nn.Linear(annot_vec_len, initSize), nn.Linear(initSize,lstm_len))

        # Our attention model
        self.attention = Attention(lstm_len, nb_annot_vec, annot_vec_len)
        # Embedding model, it transforms each word vector into an embedding vector
        self.embedding = nn.Embedding(wordmap_len, embedding_len)  
        # LSTM model. We use LSTMCell and implement the loop manualy to use attention
        self.lstm = nn.LSTMCell(embedding_len + annot_vec_len, lstm_len, bias=True) 
        # A simple linear layer to compute the vocabulary scores from the hidden state
        self.scoring_layer = nn.Linear(lstm_len, wordmap_len)
        self.drop_out = nn.Dropout(self.dropout_p)
        
    def init_hidden_state(self, image_features, device):
        """
        Initialize the hidden state and cell state for each sentence.
        Using zero tensors for now, might change later.

        Params:
            - image_features : vector of image features (batch_size,14,14,512)

        Output:
            - Hidden state : vector for initial hidden state
            - Cell state : vector for initial cell state
        """
        batch_size = image_features.size(0)
        flatten_features = image_features.view(batch_size,-1,self.annot_vec_len)
        mean_features = flatten_features.mean(dim=1)
        
        h = self.init_h_MLP(mean_features)
        c = self.init_c_MLP(mean_features)

        return h, c
    
    
    def initCaption(self, SOS_token, batch_size = 1):
        word_input = SOS_token*torch.ones(batch_size, device=device, dtype = torch.long)
        return word_input
    
    

    def forward(self, image_features, input_word, h, c):
        """
        Forward propagation.

        Params:
            - image_features : vector of image features (batch_size,14,14,512)
            - caps: image captions (batch_size, max_caption_length)
            - caplens: image caption lengths (batch_size, 1)
        Output:
            - scores : scores for each word (batch_size, wordmap_len)
            - caps_sorted : a sorted list of caps by lenghts.
            - decode lengths : caplens - 1
            - weights : attention weights
            - sort indices : can be used later
        """
        batch_size = image_features.size(0)
        # Flatten image 14*14 -> 196
        features = image_features.view(batch_size, -1, self.annot_vec_len)
               
        # At the moment t we only decode the sentences that haven't reached <end>, so the K first sentences
        # Since they are sorted by lenght we can just use [:K]     
        output_word = input_word.detach()
        new_h, new_c = h.clone(), c.clone()
        
        # We first generate the attention weighted images. Alpha is the weights of the attention model.
        embeddings = self.embedding(input_word) 

        attention_encoding, weights = self.attention(features, h)
 
        # Concatenate Previous word + features
        lstm_input = torch.cat([embeddings, attention_encoding], dim=1) 
        
        # We run the LSTM cell using the decode imput and (hidden,cell) states
        new_h, new_c = self.lstm(lstm_input, (h,c)) 
        new_h, new_c = self.drop_out(new_h), self.drop_out(new_c)
        # The hidden state is transformed into vocabulary scores by a simple linear layer
        scores = self.scoring_layer(new_h)  # (k, wordmap_len)
        output_word = scores.argmax(dim=1) 
        
        return scores, output_word, new_h, new_c, weights





def decode(decoder, image_features, caps, caplens, max_cap_len, word_map, teacher_forcing, training = True):
    if training:
        batch_size = image_features.size(0)
        nb_annotation_vec = image_features.view(batch_size, -1, decoder.annot_vec_len).size(1)
        
        EOS_token = word_map['<end>']
        SOS_token = word_map['<start>']
        output_word = decoder.initCaption(SOS_token, batch_size)
        decode_lengths = (max_cap_len + 1)*torch.ones(batch_size, dtype=torch.long, device=device)
        h, c = decoder.init_hidden_state(image_features,device)
    
        scores = torch.empty(batch_size, max_cap_len + 2, decoder.wordmap_len, device=device,requires_grad=True)
        scores[:,0,SOS_token] = 1.
        scores[:,max_cap_len+1,EOS_token] = 1.
        attn_weights = torch.empty(batch_size, max_cap_len + 2, nb_annotation_vec, device=device,requires_grad=True)
    
        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        if not teacher_forcing:
            for t in range(1,max_cap_len + 1):
                score, word, h, c, weights = decoder(image_features, output_word, h, c)
                # Update score and attention_weight values at time t
                scores[:,t,:] = score.clone()
                attn_weights[:,t,:] = weights.clone()
                output_word = word
                
                for i in range(batch_size):
                    if output_word[i] == EOS_token:
                        decode_lengths[i] = t
                    
                    
        else:  # teacher_forcing    
            for t in range(1, int(caplens.max().item())):
                score, word, h, c, weights = decoder(image_features, output_word, h, c)
   
                # Update score and attention_weight values at time t
                scores[:,t,:] = score.clone()
                attn_weights[:,t,:] = weights.clone()
                output_word = word
                
                for i in range(batch_size):
                    if output_word[i] == EOS_token:
                        decode_lengths[i] = t
                        
    else:   # Not training
        with torch.no_grad():
            batch_size = image_features.size(0)
            nb_annotation_vec = image_features.view(batch_size, -1, decoder.annot_vec_len).size(1)
            EOS_token = word_map['<end>']
            SOS_token = word_map['<start>']
            output_word = decoder.initCaption(SOS_token, batch_size)
            decode_lengths = (max_cap_len + 1)*torch.ones(batch_size, dtype=torch.long, device=device)
            h, c = decoder.init_hidden_state(image_features,device)
        
            scores = torch.zeros(batch_size, max_cap_len + 2, decoder.wordmap_len, device=device)
            scores[:,0,SOS_token] = 1.
            scores[:,max_cap_len+1,EOS_token] = 1.
            attn_weights = torch.zeros(batch_size, max_cap_len + 2, nb_annotation_vec, device=device)
        
            # At each time-step, decode by
            # attention-weighing the encoder's output based on the decoder's previous hidden state output
            # then generate a new word in the decoder with the previous word and the attention weighted encoding
            for t in range(1,max_cap_len + 1):
                score, word, h, c, weights = decoder(image_features, output_word, h, c)
                # Update score and attention_weight values at time t
                scores[:,t,:] = score.clone()
                attn_weights[:,t,:] = weights.clone()
                output_word = word
                
                for i in range(batch_size):
                    if output_word[i] == EOS_token:
                        decode_lengths[i] = t
                        
    return scores, caps, caplens, decode_lengths, attn_weights