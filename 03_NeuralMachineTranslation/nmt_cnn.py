# CNNEncoder from torchnlp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Conv1d, Linear, ReLU

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import random
import math
import time


class CNNEncoder(torch.nn.Module):
    """ A combination of multiple convolution layers and max pooling layers.

    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filters. The number of times a convolution layer will be used
    is ``num_tokens - ngram_size + 1``. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max.

    This operation is repeated for every ngram size passed, and consequently the dimensionality of
    the output after maxpooling is ``len(ngram_filter_sizes) * num_filters``.  This then gets
    (optionally) projected down to a lower dimensional output, specified by ``output_dim``.

    We then use a fully connected layer to project in back to the desired output_dim.  For more
    details, refer to "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
    Networks for Sentence Classification", Zhang and Wallace 2016, particularly Figure 1.

    **Thank you** to AI2 for their initial implementation of :class:`CNNEncoder`. Here is
    their `License
    <https://github.com/allenai/allennlp/blob/master/LICENSE>`__.

    Args:
        embedding_dim (int): This is the input dimension to the encoder.  We need this because we
          can't do shape inference in pytorch, and we need to know what size filters to construct
          in the CNN.
        num_filters (int): This is the output dim for each convolutional layer, which is the number
          of "filters" learned by that layer.
        ngram_filter_sizes (:class:`tuple` of :class:`int`, optional): This specifies both the
          number of convolutional layers we will create and their sizes. The default of
          ``(2, 3, 4, 5)`` will have four convolutional layers, corresponding to encoding ngrams of
          size 2 to 5 with some number of filters.
        conv_layer_activation (torch.nn.Module, optional): Activation to use after the convolution
          layers.
        output_dim (int or None, optional) : After doing convolutions and pooling, we'll project the
          collected features into a vector of this size.  If this value is ``None``, we will just
          return the result of the max pooling, giving an output of shape
          ``len(ngram_filter_sizes) * num_filters``.
    """

    def __init__(self,
                 embedding_dim,
                 num_filters,
                 ngram_filter_sizes=(2, 3, 4, 5),
                 conv_layer_activation=ReLU(),
                 output_dim=None):
        super(CNNEncoder, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_filters = num_filters
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = conv_layer_activation
        self._output_dim = output_dim

        self._convolution_layers = [
            Conv1d(
                in_channels=self._embedding_dim,
                out_channels=self._num_filters,
                kernel_size=ngram_size) for ngram_size in self._ngram_filter_sizes
        ]
        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module('conv_layer_%d' % i, conv_layer)

        maxpool_output_dim = self._num_filters * len(self._ngram_filter_sizes)
        if self._output_dim:
            self.projection_layer = Linear(maxpool_output_dim, self._output_dim)
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    def get_input_dim(self):
        return self._embedding_dim

    def get_output_dim(self):
        return self._output_dim

    def forward(self, tokens, mask=None):
        """
        Args:
            tokens (:class:`torch.FloatTensor` [batch_size, num_tokens, input_dim]): Sequence
                matrix to encode.
            mask (:class:`torch.FloatTensor`): Broadcastable matrix to `tokens` used as a mask.
        Returns:
            (:class:`torch.FloatTensor` [batch_size, output_dim]): Encoding of sequence.
        """
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()

        # Our input is expected to have shape `(batch_size, num_tokens, embedding_dim)`.  The
        # convolution layers expect input of shape `(batch_size, in_channels, sequence_length)`,
        # where the conv layer `in_channels` is our `embedding_dim`.  We thus need to transpose the
        # tensor first.
        tokens = torch.transpose(tokens, 1, 2)
        # Each convolution layer returns output of size `(batch_size, num_filters, pool_length)`,
        # where `pool_length = num_tokens - ngram_size + 1`.  We then do an activation function,
        # then do max pooling over each filter for the whole input sequence.  Because our max
        # pooling is simple, we just use `torch.max`.  The resultant tensor of has shape
        # `(batch_size, num_conv_layers * num_filters)`, which then gets projected using the
        # projection layer, if requested.

        filter_outputs = []
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, 'conv_layer_{}'.format(i))
            filter_outputs.append(self._activation(convolution_layer(tokens)).max(dim=2)[0])

        # Now we have a list of `num_conv_layers` tensors of shape `(batch_size, num_filters)`.
        # Concatenating them gives us a tensor of shape
        # `(batch_size, num_filters * num_conv_layers)`.
        maxpool_output = torch.cat(
            filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]

        if self.projection_layer:
            result = self.projection_layer(maxpool_output)
        else:
            result = maxpool_output
        return result

 
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, num_filters, ngram_filter_sizes, output_dim, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim
        )
        
        self.cnn = CNNEncoder(
            embedding_dim=emb_dim,
            num_filters=num_filters,
            ngram_filter_sizes=ngram_filter_sizes,
            conv_layer_activation=ReLU(),
            output_dim=output_dim
        )        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):
        
        #src = [src sent len, batch size]
        embedded = self.embedding(src)        
        embedded = self.dropout(embedded)        
        output = self.cnn(embedded.permute(1,0,2))        
        return output
    

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )
        
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )
       
        self.out = nn.Linear(
            in_features=hid_dim,
            out_features=output_dim
        )
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, input, hidden, cell):
       
        input = input.unsqueeze(0)        
        #input = [1, batch size]

        embedded = self.dropout(self.embedding(input))     
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.out(output.squeeze(0))
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
                
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        output = self.encoder(src)        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        part4 = self.decoder.hid_dim
        output, hidden, cell = self.decoder(input, output[:,:part4*2].contiguous().view(2,batch_size, part4),output[:,part4*2:].contiguous().view(2,batch_size, part4))
        outputs[1] = output
        teacher_force = random.random() < teacher_forcing_ratio
        top1 = output.max(1)[1]
        input = (trg[1] if teacher_force else top1)
        
        for t in range(2, max_len):
            
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs
