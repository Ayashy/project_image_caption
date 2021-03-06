import torch
from torch import nn
import torchvision
from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np
from attention import Attention

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class Decoder(nn.Module):
    """
    Implements the decoder model.
    it's based on an lstm and uses attention.
    """

    def __init__(self, attention_len, embedding_len, features_len, wordmap_len, lstm_len=512, dropout=0.5):
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
        self.attention_len = attention_len
        self.embedding_len = embedding_len
        self.features_len = features_len
        self.wordmap_len = wordmap_len
        self.dropout = dropout

        # Linear layers to initialize states
        self.init_h = nn.Linear(features_len, lstm_len)
        self.init_c = nn.Linear(features_len, lstm_len)
        # Our attention model
        self.attention = Attention(lstm_len, attention_len, features_len)
        # Embedding model, it transforms each word vector into an embedding vector
        self.embedding = nn.Embedding(wordmap_len, embedding_len)
        # Dropout regularisation
        self.dropout = nn.Dropout(p=self.dropout)
        # LSTM model. We use LSTMCell and implement the loop manualy to use attention
        self.lstm = nn.LSTMCell(
            embedding_len + features_len, lstm_len, bias=True)
        # A simple linear layer to compute the vocabulary scores from the hidden state
        self.scoring_layer = nn.Linear(features_len, wordmap_len)
        # Sigmoid gate to improve learning
        self.f_beta = nn.Linear(lstm_len, features_len)
        self.sigmoid = nn.Sigmoid()

        self.fine_tune_embeddings(True)

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
        mean_encoder_out = image_features.mean(dim=1)
        h = self.init_h(mean_encoder_out).to(device)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out).to(device)

        return h, c

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune
            
    def forward(self, image_features, caps, caplens, forcing=0.3):
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
        encoder_dim = image_features.size(-1)

        # Flatten image 14*14 -> 196
        image_features = image_features.view(batch_size, -1, encoder_dim)
        num_pixels = image_features.size(1)

        # Sort the sentences by decreasing lenght, so we can decode only the k first sentences that haven't
        # reached <end> yet. We can use index to select them, but this is cleaner
        caplens, sort_ind = caplens.squeeze(1).sort(dim=0, descending=True)
        image_features = image_features[sort_ind]
        caps = caps[sort_ind]

        # Transforms the captions into embedding vectors
        embeddings = self.embedding(caps)
        # Initialize LSTM  hidden and cell state
        h, c = self.init_hidden_state(image_features, device)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caplens+1).tolist()

        # Result tensors
        scores = torch.zeros(batch_size, max(decode_lengths),
                             self.wordmap_len, device=device)
        weights = torch.zeros(batch_size, max(
            decode_lengths), num_pixels, device=device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):

            # The last word generated
            previous_words=None
            # At the moment t we only decode the sentences that haven't reached <end>, so the K first sentences
            # Since they are sorted by lenght we can just use [:K]
            k = sum([l > t for l in decode_lengths])

            if t==0:
                previous_words=embeddings[:k, 0, :]
            elif  torch.rand(1)<forcing:
                previous_words=embeddings[:k, t-1, :]
            else:
                previous_words=self.embedding(torch.argmax(scores[:k, t-1],dim=1))


            # We first generate the attention weighted images. Alpha is the weights of the attention model.
            attention_encoding, alpha = self.attention(
                image_features[:k], h[:k])

            #gate = self.sigmoid(self.f_beta(h[:k]))  # gating scalar, (batch_size_t, encoder_dim)
            #attention_encoding = gate * attention_encoding

            # Concatenate Previous word + features
            decode_input = torch.cat(
                [previous_words, attention_encoding], dim=1)

            # We run the LSTM cell using the decode imput and (hidden,cell) states
            h, c = self.lstm(decode_input, (h[:k], c[:k]))

            # The hidden state is transformed into vocabulary scores by a simple linear layer
            score = self.scoring_layer(h)  # (k, wordmap_len)

            # Finaly we store the scores and weights
            scores[:k, t, :] = score
            weights[:k, t, :] = alpha

        return scores, caps, decode_lengths, weights, sort_ind
