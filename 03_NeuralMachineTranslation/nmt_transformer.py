import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import random
import math
import time

 
class Seq2Seq(nn.Module):
    def __init__(self, src_dim, trg_dim, emb_dim, dropout, device):
        super().__init__()       
        self.device = device
        self.trg_dim = trg_dim
        #self.dropout = nn.Dropout(p=dropout)
        
        self.embedding_src = nn.Embedding(
            num_embeddings=src_dim,
            embedding_dim=emb_dim
        )
        
        self.transformer = torch.nn.Transformer(d_model=emb_dim, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=dropout, activation='relu', custom_encoder=None, custom_decoder=None)
        
        self.embedding_trg = nn.Embedding(
            num_embeddings=trg_dim,
            embedding_dim=emb_dim
        )
           
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.trg_dim
        
        #embedded_src = [S, N, E]
        embedded_src = self.embedding_src(src)    
        #embedded_src = self.dropout(embedded_src)
        
        
        #input = input.unsqueeze(0)        
        #input = [1, batch size]        
        # Compute an embedding from the input data and apply dropout to it
        embedded_trg = self.embedding_trg(trg)
        #embedded_trg= self.dropout(embedded_trg)               
        output = self.transformer(embedded_src, embedded_trg)
        
        # src: (S, N, E)
        # tgt: (T, N, E)
        # S is the source sequence length,
        # T is the target sequence length, 
        # N is the batch size,
        # E is the feature number. Equal d_model Transformer        
        
        #tensor to store decoder outputs
        #outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #first input to the decoder is the <sos> tokens
        #input = embedded_trg[0,:,:]
        
        #for t in range(1, max_len):
            
         #   output, hidden, cell = self.decoder(input, hidden, cell)
         #   outputs[t] = output
         #   teacher_force = random.random() < teacher_forcing_ratio
         #   top1 = output.max(1)[1]
         #   input = (trg[t] if teacher_force else top1)
        
        return outputs
