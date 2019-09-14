import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(1000, hid_dim)
        
        self.layers = nn.ModuleList([decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
                                     for _ in range(n_layers)])
        
        self.fc = nn.Linear(hid_dim, output_dim)
        
        self.do = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, src, trg_mask, src_mask):
        
        #trg = [batch_size, trg sent len]
        #src = [batch_size, src sent len]
        #trg_mask = [batch size, trg sent len]
        #src_mask = [batch size, src sent len]
        
        pos = torch.arange(0, trg.shape[1]).unsqueeze(0).repeat(trg.shape[0], 1).to(self.device)
                
        trg = self.do((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        
        #trg = [batch size, trg sent len, hid dim]
        
        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)
            
        return self.fc(trg)

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()
        
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)
        
    def forward(self, trg, src, trg_mask, src_mask):
        
        #trg = [batch size, trg sent len, hid dim]
        #src = [batch size, src sent len, hid dim]
        #trg_mask = [batch size, trg sent len]
        #src_mask = [batch size, src sent len]
                
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
                
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))
        
        trg = self.ln(trg + self.do(self.pf(trg)))
        
        return trg