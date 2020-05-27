# not finished
# want realized NMT with CNN:
# A Convolutional Encoder Model for Neural Machine Translation" 
# Jonas Gehring, Michael Auli, David Grangier, Yann N. Dauphin
# Facebook AI Research

import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import random
import math
import time


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()      
       
        
    def forward(self, src):        
        pass
    

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        
    def forward(self, input, hidden, cell):
        pass


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device


        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        pass
