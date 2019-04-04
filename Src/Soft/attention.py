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

        self.encoder_att = nn.Linear(features_len, attention_len)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(lstm_len, attention_len)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_len, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

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

        att1 = self.encoder_att(image_features)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(hidden_state)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (image_features * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha

