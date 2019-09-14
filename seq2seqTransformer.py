import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
from transformer_model.seq2seq import Seq2Seq
from transformer_model.encoder import Encoder, EncoderLayer
from transformer_model.decoder import Decoder, DecoderLayer
from transformer_model.self_attention import SelfAttention
from transformer_model.evaluation import evaluate, calculate_bleu_score, epoch_time, translate
from transformer_model.positionwise_feed_forward import PositionwiseFeedforward
from transformer_model.train import train
from transformer_model.tokenization import tokenize

import random
import math
import os
import time
import reinforcement_learning.reinforce as rl

SEED = 1

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
PRE_TRAINING_ENABLED = False
PATH = 'data_new/' if PRE_TRAINING_ENABLED else 'data_new/gold/'
BASE_DIR = os.path.join( os.path.dirname( __file__ ))

#spacy_de = spacy.load('de')

SRC = Field(tokenize=tokenize, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
TRG = Field(tokenize=tokenize, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)

transl_dataset = TranslationDataset('data_new/', ('conala-all-data.intent', 'conala-all-data.snippet'), fields=(SRC, TRG))

intent_datasets_path = '_conala.intent' if PRE_TRAINING_ENABLED else '_conala2k.intent'
snippet_datasets_path = '_conala.snippet' if PRE_TRAINING_ENABLED else '_conala2k.snippet'  
  
train_data, valid_data, test_data = transl_dataset.splits((intent_datasets_path, snippet_datasets_path), 
                                                          fields=(SRC, TRG), path=PATH)
SRC.build_vocab(train_data, max_size=4000)
TRG.build_vocab(train_data, max_size=4000)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
     batch_size=BATCH_SIZE,
     device=DEVICE)

def create_model(SRC):
    input_dim = len(SRC.vocab)
    hid_dim = 256
    n_layers = 1 # 6
    n_heads = 8
    pf_dim = 512 # 2048
    dropout = 0.2

    enc = Encoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, EncoderLayer, SelfAttention, PositionwiseFeedforward, dropout, DEVICE)

    output_dim = len(TRG.vocab)
    hid_dim = 256
    n_layers = 1 # 6
    n_heads = 8
    pf_dim = 256 # 2048
    dropout = 0.2

    dec = Decoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, DEVICE)

    pad_idx = SRC.vocab.stoi['<pad>']

    model = Seq2Seq(enc, dec, pad_idx, DEVICE).to(DEVICE)

    return model, pad_idx 

model, pad_idx = create_model(SRC)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'transformer-seq2seq.pt')

def start_train_model(n_epochs):
    CLIP = 1

    best_valid_loss = float('inf')

    if not os.path.isdir(f'{SAVE_DIR}'):
        os.makedirs(f'{SAVE_DIR}')

    for epoch in range(n_epochs):
        
        start_time = time.time()
        
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        
        #rl_loss = rl.run_reinforce(model, train_iterator)
        test_bleu_score = calculate_bleu_score(model, test_data, SRC, TRG, BASE_DIR)
        print(f'|Test BLEU: {test_bleu_score} Epoch: {epoch+1:03} | Time: {epoch_mins}m {epoch_secs}s| Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')

#------------------------train + validaion --------#
N_EPOCHS = 100
start_train_model(N_EPOCHS)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))

test_loss = evaluate(model, test_iterator, criterion)
bleu_score = calculate_bleu_score(model, test_data, SRC, TRG, BASE_DIR)

print(f'Test BLEU: {bleu_score}| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} ')