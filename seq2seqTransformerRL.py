import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys 

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
from transformer_model.seq2seq import Seq2Seq
from transformer_model.encoder import Encoder, EncoderLayer
from transformer_model.decoder import Decoder, DecoderLayer
from transformer_model.self_attention import SelfAttention
from transformer_model.evaluation import evaluate, calculate_bleu_score, calculate_bleu_score_batch, epoch_time, translate
from transformer_model.positionwise_feed_forward import PositionwiseFeedforward
from transformer_model.train import train
from transformer_model.tokenization import tokenize
import argparse

import random
import math
import os
import time
import reinforcement_learning.rl_reinforce as rl
import numpy as np 

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--num_epochs', type=int, default=5, metavar='N', help='number of epochs to run')
parser.add_argument('--epochs_rl', type=int, default=100, metavar='N', help='epocs to start rl')
parser.add_argument('--local', type=int, default=1, metavar='N', help='local running')
parser.add_argument('--beam_size', type=int, default=3, metavar='N', help='beam')
parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='batch size')
parser.add_argument('--testing', type=int, default=0, metavar='N', help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, metavar='N', help='batch size')

args = parser.parse_args(args=sys.argv[1:])

print('Step 0')

SEED = 1
RL_EPOCHS = 1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
PRE_TRAINING_ENABLED = False
PATH = 'data_new/' if PRE_TRAINING_ENABLED else 'data_new/gold/'
BASE_DIR = '/'

print('Step 1')

#spacy_de = spacy.load('de')

SRC = Field(tokenize=tokenize, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
TRG = Field(tokenize=tokenize, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)

transl_dataset = TranslationDataset('data_new/', ('conala-all-data.intent', 'conala-all-data.snippet'), fields=(SRC, TRG))

intent_datasets_path = '_conala.intent' if PRE_TRAINING_ENABLED else '_conala2k.intent'
snippet_datasets_path = '_conala.snippet' if PRE_TRAINING_ENABLED else '_conala2k.snippet'  

print('Data fetched')	

train_data, valid_data, test_data = transl_dataset.splits((intent_datasets_path, snippet_datasets_path), 
                                                          fields=(SRC, TRG), path=PATH)
SRC.build_vocab(train_data, max_size=4000)
TRG.build_vocab(train_data, max_size=4000)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Data Splitted, using cuda: {}'.format(torch.cuda.is_available()))	

train_data.examples = train_data.examples[0:50]

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
     batch_size=args.batch_size,
     device=DEVICE)

train_iterator.shuffle = False

def create_model(SRC, local = False):
    input_dim = len(SRC.vocab)
    hid_dim = 256
    n_layers = 1 # 6
    n_heads = 8
    pf_dim = 512 # 2048
    dropout = 0.2

    if local:
        input_dim = len(SRC.vocab)
        hid_dim = 64
        n_layers = 1 # 6
        n_heads = 1
        pf_dim = 256 # 2048
        dropout = 0.2

    enc = Encoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, EncoderLayer, SelfAttention, PositionwiseFeedforward, dropout, DEVICE)
    
    output_dim = len(TRG.vocab)
    hid_dim = 256
    n_layers = 1 # 6
    n_heads = 8
    pf_dim = 256 # 2048
    dropout = 0.2

    if local:
        output_dim = len(TRG.vocab)
        hid_dim = 64
        n_layers = 1 # 6
        n_heads = 1
        pf_dim = 256 # 2048
        dropout = 0.2

    dec = Decoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, DEVICE)

    pad_idx = SRC.vocab.stoi['<pad>']

    model = Seq2Seq(enc, dec, pad_idx, DEVICE).to(DEVICE)

    return model, pad_idx 

def create_transformer_model():
    print('Creating model')	
    args.local = True if args.local == 1 else False
    model, pad_idx = create_model(SRC, args.local)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    return model, optimizer, criterion

def start_train_model(args, model_save_path, SAVE_DIR, model, optimizer, criterion, is_code_reward_enabled):
    CLIP = 1
    n_epochs = args.num_epochs
    rewards_per_epoch = []

    best_valid_loss = float('inf')

    if not os.path.isdir(f'{SAVE_DIR}'):
        os.makedirs(f'{SAVE_DIR}')
    
    for epoch in range(n_epochs):
        is_rl_enabled = epoch > args.epochs_rl

        if is_rl_enabled:
            prediction_log_file = open('logs' + '/{}real_rl_predictions_epoch_{}.txt'.format('codeReward' if is_code_reward_enabled else 'bleu', 
            str(epoch)), "w", encoding="utf-8")
        else:
            prediction_log_file = open('logs/training' + '/trainingBigModel_epoch_{}.txt'.format(str(epoch)), "w", encoding="utf-8")

        start_time = time.time()
        
        train_loss, rewards = train(model, train_iterator, optimizer, criterion, CLIP, SRC, TRG, is_rl_enabled, args.beam_size, is_code_reward_enabled, SAVE_DIR, prediction_log_file)
        rewards_per_epoch.append(rewards)
        
        valid_loss = evaluate(model, valid_iterator, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_save_path)

        if is_rl_enabled and epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'transformer-seq2seq-rl-{}.pt'.format(epoch)))

        if(epoch % 10 == 0 or is_rl_enabled):
            valid_bleu_score = calculate_bleu_score_batch(model, valid_iterator, SRC, TRG, BASE_DIR)
            #test_bleu_score = 0
            #print(f'Epoch: {epoch+1:03} | Test BLEU: {test_bleu_score} | Time: {epoch_mins}m {epoch_secs}s| Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')
            print(f'Epoch: {epoch+1:03} | Valid BLEU: {valid_bleu_score} | Time: {epoch_mins}m {epoch_secs}s| Train Loss: {train_loss:.3f} | Train PPL: {train_loss:7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {valid_loss:7.3f} |')
        else:
            print(f'|Epoch: {epoch+1:03} | Time: {epoch_mins}m {epoch_secs}s| Train Loss: {train_loss:.3f} | Train PPL: {train_loss:7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {valid_loss:7.3f} |') 

        prediction_log_file.close()

    return rewards_per_epoch

#------------------------train + validaion --------#
def run_rl(is_code_reward_enabled):
    SAVE_DIR = 'models'
    print('Loading model and initial loss is:')
    #torch.backends.cudnn.enabled = False
    if(args.local == 1):
        model_save_path = os.path.join(SAVE_DIR, 'transformer-seq2seq-15-epochs.pt')
    else: 
        model_save_path = os.path.join(SAVE_DIR, 'transformer-seq2seq-BIG-epochs.pt')
    # loading prev model
    model, optimizer, criterion = create_transformer_model()

    if (torch.cuda.is_available()):
        model.load_state_dict(torch.load(model_save_path))
    else:
        model.load_state_dict(torch.load(model_save_path, map_location='cpu'))

    print('Starting RL training')
    args.epochs_rl = -1
    model_save_path = os.path.join(SAVE_DIR, 'transformer-seq2seq_with_rl.pt')
    rewards_per_epoch = start_train_model(args, model_save_path, SAVE_DIR, model, optimizer, criterion, is_code_reward_enabled)
    np.save(os.path.join(SAVE_DIR, 'rewards_per_epoch{}.npy'.format( 
        '_code_reward' if is_code_reward_enabled else '_bleu_reward')),
         rewards_per_epoch)

    print('***Finished Training Model****')
    test_loss = evaluate(model, test_iterator, criterion)
    bleu_score = calculate_bleu_score(model, test_data, SRC, TRG, BASE_DIR)
    #print(f'Test BLEU: {bleu_score}| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} ')
    print(f'Test BLEU: {bleu_score}| Test Loss: {test_loss:.3f} | Test PPL: {test_loss:7.3f} ')

def run_train_from_saved_model():
    SAVE_DIR = 'models'
    print('Loading model and initial loss is:')
    #torch.backends.cudnn.enabled = False
    if(args.local == 1):
        model_save_path = os.path.join(SAVE_DIR, 'transformer-seq2seq-15-epochs.pt')
    else: 
        model_save_path = os.path.join(SAVE_DIR, 'transformer-seq2seq-BIG-epochs.pt')
    # loading prev model
    model, optimizer, criterion = create_transformer_model()
    if (torch.cuda.is_available()):
        model.load_state_dict(torch.load(model_save_path))
    else:
        model.load_state_dict(torch.load(model_save_path,  map_location='cpu'))

    print('Starting training without RL')
    args.epochs_rl = 100
    model_save_path = os.path.join(SAVE_DIR, 'transformer-seq2seq_with_test.pt')
    is_code_reward_enabled = False
    rewards_per_epoch = start_train_model(args, model_save_path, SAVE_DIR, model, optimizer,criterion, is_code_reward_enabled)

    print('***Finished Training Model****')
    test_loss = evaluate(model, test_iterator, criterion)
    bleu_score = calculate_bleu_score(model, test_data, SRC, TRG, BASE_DIR)
    print(f'Test BLEU: {bleu_score}| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} ')

def run_train_new_model():
    SAVE_DIR = 'models'
    print('Loading model and initial loss is:')
    #torch.backends.cudnn.enabled = False
    model_save_path = os.path.join(SAVE_DIR, 'transformer-seq2seq-BIG-epochs.pt')
    # loading prev model
    model, optimizer, criterion = create_transformer_model()

    print('Starting training new model')
    args.epochs_rl = 200
    is_code_reward_enabled = False    
    rewards_per_epoch = start_train_model(args, model_save_path, SAVE_DIR, model, optimizer,criterion, is_code_reward_enabled)

    print('***Finished Training Model****')
    torch.save(model.state_dict(), model_save_path)
    test_loss = evaluate(model, test_iterator, criterion)
    bleu_score = calculate_bleu_score(model, test_data, SRC, TRG, BASE_DIR)
    print(f'Test BLEU: {bleu_score}| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} ')

if __name__ == "__main__":
    print('testing', args.testing)
    args.testing = 3
    args.local = 1
    #args.num_epochs = 20
    #args.lr = 0.000001

    if(args.testing == 0):
        run_train_from_saved_model()
    elif (args.testing == 1):
        print('Starting RL: training with code reward')
        use_code_reward = True
        run_rl(use_code_reward)
    elif (args.testing == 2):
        print('Starting RL: training WITHOUT code reward')
        use_code_reward = False
        run_rl(use_code_reward)  
    elif (args.testing == 3):
        print('Starting Training new model')
        use_code_reward = False
        run_train_new_model()  