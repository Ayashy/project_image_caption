import torch
from torch import nn
import torchvision
from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np

class Attention(nn.Module):
    """
    Implements the attention model. 
    It is based on 1 linear layer folowed by tanh then a softmax to compute the weights,
    Then outputs the features*weights , weights.
    """

    def __init__(self, lstm_len, nb_annot_vec=256, annot_vec_len=512):
        """
        Params:
            - features_len : size of the images feature vector we're using 512
            - lstm_len : size of the lstm network (it's the size of the hidden and cell states also)
            - attention_len : size of the attention network (Not sure how it affects the learning yet)
        """
        super(Attention, self).__init__()

        # Simple linear layer to project the encoder features
        self.image_layer = nn.Linear(annot_vec_len, 1, bias=False) 
        # Simple linear layer to project the hidden state 
        self.hidden_layer = nn.Linear(lstm_len, nb_annot_vec, bias=False)  
        # A Tanh layer
        self.tanh = nn.Tanh()
        
        self.second_layer = nn.Linear(nb_annot_vec, nb_annot_vec, bias=False)
        
        # Softmax layer to compure weights
        self.softmax = nn.Softmax(dim=1)  

    def forward(self, image_features, hidden_state): 
        """
        Forward propagation. Computes the attention weights using the equation

                        Si=tanh(Wc*C+Wx*Xi) , weights=softmax(si)

        Params:
            - image_features : vector of image features (batch_size,14x14,512)
            - hidden_state : vector Ht-1 previous hidden state (batch_size,lstm_len)

        Output:
            - attention_features : weights*features (batchsize,lstm_len)
            - attention_weights : weights computed with softmax (batchsize,nb_image_sections=14*14)
        """
        
        batch_size, features_len = image_features.size(0), image_features.size(2)
        flatten_features = image_features.view(batch_size,-1,features_len)   # batch_size x (16x16) x 512
        features = self.image_layer(flatten_features).squeeze(2)             # batch_size x (16x16)
        
        hidden = self.hidden_layer(hidden_state)                             # batch_size x (16x16)
        
        merged = features + hidden
        tanh = self.tanh(merged)

        combined = self.second_layer(tanh)
        tanh = self.tanh(combined)
        
        attention_weights = self.softmax(tanh)                               # batch_size x (16x16)
        attention_features = (image_features * attention_weights.unsqueeze(2)).sum(dim=1)  # Sum over spatial dimension
                                                                             # batch_size x 512 
        return attention_features, attention_weights

