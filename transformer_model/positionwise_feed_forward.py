import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)
        
        self.do = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, sent len, hid dim]
        
        x = x.permute(0, 2, 1)
        
        #x = [batch size, hid dim, sent len]
        
        x = self.do(F.relu(self.fc_1(x)))
        
        #x = [batch size, ff dim, sent len]
        
        x = self.fc_2(x)
        
        #x = [batch size, hid dim, sent len]
        
        x = x.permute(0, 2, 1)
        
        #x = [batch size, sent len, hid dim]
        
        return x