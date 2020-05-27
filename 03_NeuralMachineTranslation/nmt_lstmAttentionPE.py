# Positional encoding realized by https://github.com/kaushalshetty

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import numpy as np

import random
import math
import time


class Attention(nn.Module):
    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()


    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights

    

def position_encoding_init(n_position, emb_dim):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
    

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # apply sin on 0th,2nd,4th...emb_dim
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # apply cos on 1st,3rd,5th...emb_dim
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


class PositionalEncoder(torch.nn.Module):
    """
    Sets up embedding layer for word sequences as well as for word positions.Both the layers are trainable.
    Returns embeddings of words which also contains the position(time) component
    """
    def __init__(self,vocab_size,emb_dim, max_len, batch_size):    
        
        """
        Args:
            vocab_size  : [int] vocabulary size
            emb_dim     : [int] embedding dimension for words
            max_len     : [int] maxlen of input sentence
 
        Returns:
            position encoded word embeddings
 
        Raises:
            nothing
        """    
        super(PositionalEncoder,self).__init__()
        self.emb_dim = emb_dim
        
        self.src_word_emb = torch.nn.Embedding(            
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=0
        )
        
        n_position = max_len+1
        self.word_pos = Variable(torch.from_numpy(np.stack((np.arange(n_position) for i in range(batch_size)),axis=0)).type(torch.LongTensor))        
        
        self.position_enc = torch.nn.Embedding(n_position, emb_dim, padding_idx=0)
        self.position_enc.weight.data = position_encoding_init(n_position, emb_dim)        
        
        

        
    def forward(self,word_seq):
        #word_seq = [word_seq sent len, batch size]
        #word_embeddings = [src sent len, batch size, emb dim]
        #word_pos = [src sent len, batch size]    
        word_embeddings = self.src_word_emb(word_seq)
        
        seq_len = word_seq.shape[0]     
        word_pos_encoded = word_embeddings + self.position_enc(self.word_pos).permute(1,0,2)[0:seq_len]
        
        return word_pos_encoded    

    
  


    
 
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, max_len, batch_size, hid_dim, n_layers, dropout):
        super().__init__()        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embeddingPE = PositionalEncoder(
            vocab_size=input_dim,
            emb_dim = emb_dim,
            max_len = max_len,
            batch_size=batch_size
        )       
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):        
        #src = [src sent len, batch size]
        #embedded = [src sent len, batch size, emb dim]
        embedded = self.embeddingPE(src)
        #embedded = self.embeddingPE(src.transpose(0, 1)).transpose(0, 1)
        embedded = self.dropout(embedded)
        
        #output = [src sent len, batch size, hid dim * n directions (feature from each layer)]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        output, (hidden, cell) = self.rnn(embedded)
              
        return output, hidden, cell
    

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, max_len,batch_size, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embeddingPE = PositionalEncoder(
            vocab_size=output_dim,
            emb_dim = emb_dim,
            max_len = max_len,
            batch_size = batch_size
        )   
        
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )
        self.attention = Attention(self.hid_dim)
        self.out = nn.Linear(
            in_features=hid_dim,
            out_features=output_dim
        )        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, input, context_encoder, hidden, cell):        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]     
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        #embedded = [1, batch size, emb dim]
        embedded = self.embeddingPE(input)
        #embedded = self.embeddingPE(input.transpose(0, 1)).transpose(0, 1)
        embedded = self.dropout(embedded)
        
        #output = [sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        #prediction = [batch size, output dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        output_att, _ = self.attention(output.transpose(0, 1), context_encoder.transpose(0, 1))
        output_att = output_att.transpose(0, 1)
        prediction = self.out(output_att.squeeze(0))        
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        context_encoder, hidden, cell = self.encoder(src)
        input = trg[0,:]
        
        for t in range(1, max_len):
            
            output, hidden, cell = self.decoder(input,context_encoder, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs
