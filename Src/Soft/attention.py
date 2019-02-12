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

    def __init__(self, lstm_len, attention_len, features_len=512):
        """
        Params:
            - features_len : size of the images feature vector we're using 512
            - lstm_len : size of the lstm network (it's the size of the hidden and cell states also)
            - attention_len : size of the attention network (Not sure how it affects the learning yet)
        """
        super(Attention, self).__init__()

        # Simple linear layer to project the encoder features
        self.image_layer = nn.Linear(features_len, 1) 
        # Simple linear layer to project the hidden state 
        self.hidden_layer = nn.Linear(lstm_len, 1)  
        # A Tanh layer
        self.tanh = nn.Tanh()
        # Simple linear layer to merge the two vectors
        self.merge_layer = nn.Linear(attention_len, 1)  
        # Softmax layer to compure weights
        self.softmax = nn.Softmax(dim=1)  

    def forward(self, image_features, hidden_state): 
        """
        Forward propagation. Computes the attention weights using the equation

                        Si=tanh(Wc*C+Wx*Xi) , weights=softmax(si)

        Params:
            - image_features : vector of image features (batch_size,14,14,512)
            - hidden_state : vector Ht-1 previous hidden state (batch_size,lstm_len)

        Output:
            - attention_features : weights*features (batchsize,lstm_len)
            - attention_weights : weights computed with softmax (batchsize,nb_image_sections=14*14)
        """
        features = self.image_layer(image_features)  
        hidden = self.hidden_layer(hidden_state)  
        merged = features + hidden.unsqueeze(1)
        tanh = self.tanh(merged)
        attention_weights = self.softmax(tanh).squeeze(2) 
        attention_features = (image_features * attention_weights.unsqueeze(2)).sum(dim=1) 

        return attention_features, attention_weights

