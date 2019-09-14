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
from transformer_model.evaluation import evaluate, calculate_bleu_score, calculate_bleu_score_batch_real, epoch_time, translate
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
from functools import reduce

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--num_epochs', type=int, default=5, metavar='N', help='number of epochs to run')
parser.add_argument('--epochs_rl', type=int, default=100, metavar='N', help='epocs to start rl')
parser.add_argument('--local', type=int, default=1, metavar='N', help='local running')
parser.add_argument('--beam_size', type=int, default=3, metavar='N', help='beam')
parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='batch size')
parser.add_argument('--testing', type=int, default=0, metavar='N', help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, metavar='N', help='batch size')
parser.add_argument('--test_small_data', type=int, default=1, metavar='N', help='batch size')
parser.add_argument('--load_from_model', type=str, default='', metavar='N', help='batch size')
parser.add_argument('--load_from_folder', type=str, default='', metavar='N', help='load from folder')
parser.add_argument('--rl_alpha', type=float, default=0.05, metavar='N', help='rl alpha')
parser.add_argument('--seed', type=int, default=1, metavar='N', help='seed')
parser.add_argument('--pre_train_on_big_model', type=int, default=0, metavar='N', help='seed')
parser.add_argument('--rl_discount', type=float, default=0.99, metavar='N', help='rl discount')
parser.add_argument('--rl_include_baseline', type=int, default=0, metavar='N', help='include baseline')
parser.add_argument('--model_name', type=str, default='default_model', metavar='N', help='model name')
parser.add_argument('--rl_use_encoder_backprob', type=int, default=1, metavar='N', help='backprop rl')
parser.add_argument('--rl_annealing_delta', type=int, default=-1, metavar='N', help='annealing rl')
parser.add_argument('--rl_optimization_steps', type=int, default=-1, metavar='N', help='annealing rl')
parser.add_argument('--reward_type', type=int, default=4, metavar='N', help='rl code type')
parser.add_argument('--print_progress', type=int, default=0, metavar='N', help='progress print')
parser.add_argument('--start_statistics_after', type=int, default=0, metavar='N', help='statistics print')
parser.add_argument('--logging_predictions', type=int, default=0, metavar='N', help='statistics print')

#transformer 
parser.add_argument('--n_heads', type=int, default=8, metavar='N', help='statistics print')
parser.add_argument('--n_layers', type=int, default=1, metavar='N', help='statistics print')
parser.add_argument('--hid_dim', type=int, default=256, metavar='N', help='statistics print')
parser.add_argument('--vocab_size', type=int, default=4000, metavar='N', help='statistics print')

args = parser.parse_args(args=sys.argv[1:])

print('Step 0')

SEED = args.seed
RL_EPOCHS = 1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
PRE_TRAINING_ENABLED = True if  args.pre_train_on_big_model == 1 else False
PATH = 'data_new/' if PRE_TRAINING_ENABLED else 'data_new/gold/'
BASE_DIR = '/'
PLOTS_DIR = 'plots'
STATISTICS_DIR = 'statistics/all_gold_data'
SAVE_DIR = 'models'
REWARDS = 'rewards'
MODEL_NAME = args.model_name

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

SRC.build_vocab(train_data, valid_data, test_data, max_size=args.vocab_size)
TRG.build_vocab(train_data, valid_data, test_data, max_size=args.vocab_size)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Data Splitted, using cuda: {}'.format(torch.cuda.is_available()))	

if(PRE_TRAINING_ENABLED):
    train_data.examples = train_data.examples[0:150000]
    valid_data.examples = valid_data.examples[0:2500]
    print('reduced size to: 15000 train and 1500 validation samples')

if args.test_small_data == 1:
    train_data.examples = train_data.examples[0:50]
    valid_data.examples = train_data.examples[0:50]

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
     batch_size=args.batch_size,
     device=DEVICE)

test_iterator.shuffle = False

def create_model(SRC, local = False):
    input_dim = len(SRC.vocab)
    hid_dim = args.hid_dim
    n_layers = args.n_layers # 6
    n_heads = args.n_heads
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
    hid_dim = args.hid_dim
    n_layers = args.n_layers # 6
    n_heads = args.n_heads #8
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

def start_train_model(args, SAVE_DIR, MODEL_NAME_SAVE, model, optimizer, criterion, is_code_reward_enabled):
    CLIP = 0.5
    n_epochs = args.num_epochs
    rewards_per_epoch = []
    train_losses = []
    valid_losses = []
    bleu_scores = []
    best_valid_loss = float('inf')
    best_valid_bleu = 0.0
    bleu_score_old = 0.0
    test_bleu_score = 0.0
    total_mle_all_batches = []
    total_rl_all_batches = []
    stat_valid_holder = createStatisticHolder()
    stat_test_holder = createStatisticHolder()
    prediction_log_file = None 


    if not os.path.isdir(f'{SAVE_DIR}'):
        os.makedirs(f'{SAVE_DIR}')
    
    model_save_path = os.path.join(SAVE_DIR, MODEL_NAME_SAVE)

    for epoch in range(n_epochs):
        is_rl_enabled = epoch > args.epochs_rl

        if args.logging_predictions == 1:
            if is_rl_enabled:
                prediction_log_file = open('logs' + '/{}_{}_{}.txt'.format('codeReward' if is_code_reward_enabled else 'bleu', 
                MODEL_NAME_SAVE, str(epoch)), "w", encoding="utf-8")
            else:
                prediction_log_file = open('logs/training' + '/{}_{}.txt'.format(MODEL_NAME_SAVE, str(epoch)), "w", encoding="utf-8")

        start_time = time.time()
        
        include_baseline = True if args.rl_include_baseline == 1 else False 
        rl_use_encoder_backprob = True if args.rl_use_encoder_backprob == 1 else False

        sys.stdout.flush()
        train_loss, rewards, mle_epoch_batches_loss, rl_epoch_batches_loss = train(model, train_iterator, optimizer, criterion, 
            CLIP, SRC, TRG, is_rl_enabled, args.beam_size, 
            is_code_reward_enabled, SAVE_DIR, MODEL_NAME_SAVE, args.rl_discount, include_baseline, 
            args.rl_annealing_delta, args.rl_optimization_steps, args, prediction_log_file, args.rl_alpha, rl_use_encoder_backprob)

        total_mle_all_batches.append(mle_epoch_batches_loss)
        total_rl_all_batches.append(rl_epoch_batches_loss)

        rewards_per_epoch.append(rewards)
        
        valid_loss = evaluate(model, valid_iterator, criterion)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_save_path)


        if(epoch % 1 == 0 or is_rl_enabled):
            valid_bleu_score, _ = calculate_bleu_score_batch_real(model, valid_iterator, SRC, TRG, BASE_DIR)
            bleu_scores.append(valid_bleu_score)

            if ((valid_bleu_score > best_valid_bleu)):
                best_valid_bleu = valid_bleu_score

                execute_snippets = args.start_statistics_after < epoch
                try:
                    _, validation_statistics = calculate_bleu_score_batch_real(model, valid_iterator, SRC, TRG, BASE_DIR, execute_snippets = execute_snippets)
                    test_bleu_score, test_statistics = calculate_bleu_score_batch_real(model, test_iterator, SRC, TRG, BASE_DIR, execute_snippets = execute_snippets)
                    #torch.save(model.state_dict(), os.path.join(SAVE_DIR, MODEL_NAME_SAVE + f'_epoch_{epoch + 1:03}_bleu_valid_{best_valid_bleu*100:2.2f}_bleu_test_{test_bleu_score*100:2.2f}.pt'))
                    bleu_score_old = calculate_bleu_score(model, test_data, SRC, TRG, BASE_DIR)
                    update_statistics(validation_statistics, test_statistics, stat_valid_holder, stat_test_holder, epoch)

                    print(f'***** BEST VALID BLEU:{best_valid_bleu*100:2.2f} TEST BLEU:{test_bleu_score*100:2.2f}, OLD_BLEU:{bleu_score_old:03}****')
                    print(f"***** VALID executed:{validation_statistics['executed']} exec_fix:{validation_statistics['executed_with_fix']}, AST:{validation_statistics['AST']}, TOTAL: {validation_statistics['total']}****")
                    print(f"***** TEST executed:{test_statistics['executed']} exec_fix:{test_statistics['executed_with_fix']}, AST:{test_statistics['AST']}, TOTAL: {test_statistics['total']}****")
                except BaseException as e :
                    print(f'*****exeception {e} occured in validation')

            print(f'Epoch: {epoch+1:03} | Valid BLEU: {valid_bleu_score*100:.3f} | Time: {epoch_mins}m {epoch_secs}s| Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')
            #print(f'Epoch: {epoch+1:03} | Test BLEU: {test_bleu_score} | Time: {epoch_mins}m {epoch_secs}s| Train Loss: {train_loss:.3f} | Train PPL: {train_loss:7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {valid_loss:7.3f} |')
        else:
            print(f'|Epoch: {epoch+1:03} | Time: {epoch_mins}m {epoch_secs}s| Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |') 

        if args.logging_predictions == 1:
            try:
                prediction_log_file.close()
            except:
                print('prediction log file error, was already closed')
    
    save_statistic_holders(MODEL_NAME_SAVE, stat_valid_holder, stat_test_holder)
    # printing to find logs at the end
    print(f'TRAINING FINISHED: \n BEST_BLEU: {best_valid_bleu*100:2.2f} \n TEST BLEU:{test_bleu_score*100:2.2f}, \n OLD_BLEU:{bleu_score_old:03}****')
    sys.stdout.flush()
    return rewards_per_epoch, train_losses, valid_losses, bleu_scores, total_mle_all_batches, total_rl_all_batches

def createStatisticHolder():
    return { 'epoch' : [], 'executed': [], 'executed_with_fix': [], 'AST': [], 'total': [] }

#------------------------train + validaion --------# 
def run_rl(is_code_reward_enabled):
    print('Loading model and initial loss is:')
    #torch.backends.cudnn.enabled = False

    model_load_path = os.path.join(SAVE_DIR, MODEL_NAME)

    if(args.load_from_model != ''):
        model_load_path =  os.path.join(SAVE_DIR, os.path.join(args.load_from_folder, 
            args.load_from_model ))

    # loading prev model
    model, optimizer, criterion = create_transformer_model()

    if (torch.cuda.is_available()):
        model.load_state_dict(torch.load(model_load_path))
    else:
        model.load_state_dict(torch.load(model_load_path, map_location='cpu'))

    print('Starting RL training')
    rewards_per_epoch, train_losses, valid_losses, valid_bleu, mle_loss_batch, rl_loss_batch = start_train_model(args, SAVE_DIR, MODEL_NAME, model, optimizer, criterion, is_code_reward_enabled)
    np.save(os.path.join(REWARDS, MODEL_NAME + '{}.npy'.format( 
        '_code_reward' if is_code_reward_enabled else '_bleu_reward')),
         rewards_per_epoch)
    
    print('***Finished Training Model****')
    bleu_score_old = calculate_bleu_score(model, test_data, SRC, TRG, BASE_DIR)
    print(f'Test BLEU Old: {bleu_score_old:03}')
    save_losses(MODEL_NAME, train_losses, valid_losses, valid_bleu, bleu_score_old, mle_loss_batch, rl_loss_batch)
  
    test_loss = evaluate(model, test_iterator, criterion)
    bleu_score, _ = calculate_bleu_score_batch_real(model, test_iterator, SRC, TRG, BASE_DIR)
    #print(f'Test BLEU: {bleu_score}| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} ')
    print(f'Test BLEU: {bleu_score*100:03}| Test Loss: {test_loss:.3f} | Test PPL: {test_loss:7.3f} ')

def run_train_from_saved_model():
    print('Loading model and initial loss is:')
    #torch.backends.cudnn.enabled = False

    model_load_path = os.path.join(SAVE_DIR, MODEL_NAME)
    
    if(args.load_from_model != ''):
        model_load_path =  os.path.join(SAVE_DIR, os.path.join(args.load_from_folder, 
            args.load_from_model ))

    # loading prev model
    model, optimizer, criterion = create_transformer_model()
    if (torch.cuda.is_available()):
        model.load_state_dict(torch.load(model_load_path))
    else:
        model.load_state_dict(torch.load(model_load_path,  map_location='cpu'))

    print('Starting training without RL')
    args.epochs_rl = 100
    is_code_reward_enabled = False

    rewards_per_epoch, train_losses, valid_losses, valid_bleu, mle_loss_batch, rl_loss_batch = start_train_model(args, SAVE_DIR, MODEL_NAME, model, optimizer,criterion, is_code_reward_enabled)

    print('***Finished Training Model****')
    bleu_score_old = calculate_bleu_score(model, test_data, SRC, TRG, BASE_DIR)
    print(f'Test BLEU Old: {bleu_score_old:03}')
    test_loss = evaluate(model, test_iterator, criterion)    
    bleu_score, _ = calculate_bleu_score_batch_real(model, test_iterator, SRC, TRG, BASE_DIR, True)
    save_losses(MODEL_NAME, train_losses, valid_losses, valid_bleu, bleu_score, mle_loss_batch, rl_loss_batch)

    print(f'Test BLEU: {bleu_score*100:03}| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} ')

def run_train_new_model(include_rl = False, is_code_reward_enabled = False):
    new_model_name =  MODEL_NAME + '_{}'.format('code' if is_code_reward_enabled else 'bleu')

    print(f'Loading model: {new_model_name}')
    #torch.backends.cudnn.enabled = False
    model_save_path = os.path.join(SAVE_DIR, new_model_name)
    # loading prev model
    model, optimizer, criterion = create_transformer_model()

    print('Starting training new model')
    if(include_rl == False):
        args.epochs_rl = 1000000
    #else:
    #args.epochs_rl = -1
    
    rewards_per_epoch, train_losses, valid_losses, valid_bleu, mle_loss_batch, rl_loss_batch = start_train_model(args, SAVE_DIR, new_model_name, model, optimizer,criterion, is_code_reward_enabled)

    np.save(os.path.join(SAVE_DIR, f'rewards_per_epoch_model_{new_model_name}.npy'),
         rewards_per_epoch)

    print('***Finished Training Model****')
    torch.save(model.state_dict(), model_save_path) 

    test_loss = evaluate(model, test_iterator, criterion)
    bleu_score = calculate_bleu_score(model, test_data, SRC, TRG, BASE_DIR)
    save_losses(new_model_name, train_losses, valid_losses, valid_bleu, bleu_score, mle_loss_batch, rl_loss_batch)

    bleu_score2, _ = calculate_bleu_score_batch_real(model, test_iterator, SRC, TRG, BASE_DIR)
    print(f'Test BLEU old: {bleu_score:03}| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} ')
    print(f'Test BLEU: {bleu_score2*100:03}| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} ')

def update_statistics(validation_statistics, test_statistics, stat_valid_holder, stat_test_holder, epoch):
    stat_valid_holder['epoch'].append(epoch) 
    stat_test_holder['epoch'].append(epoch) 

    stat_valid_holder['executed'].append(validation_statistics['executed']) 
    stat_test_holder['executed'].append(test_statistics['executed']) 

    stat_valid_holder['executed_with_fix'].append(validation_statistics['executed_with_fix']) 
    stat_test_holder['executed_with_fix'].append(test_statistics['executed_with_fix']) 

    stat_valid_holder['AST'].append(validation_statistics['AST']) 
    stat_test_holder['AST'].append(test_statistics['AST']) 

    stat_valid_holder['total'].append(validation_statistics['total']) 
    stat_test_holder['total'].append(test_statistics['total']) 

def save_statistic_holders(model_name, stat_valid_holder, stat_test_holder):
    np.save(STATISTICS_DIR + '/' + 'valid_' + model_name, stat_valid_holder)
    np.save(STATISTICS_DIR + '/' + 'test_' + model_name, stat_test_holder)

def save_losses(MODEL_NAME, train_losses, valid_losses, valid_bleu, bleu_test_loss, mle_per_batch, rl_per_batch):
    np.save(PLOTS_DIR + '/' + 'train_losses_' + MODEL_NAME, train_losses)
    np.save(PLOTS_DIR + '/' + 'valid_losses_' + MODEL_NAME, valid_losses)
    np.save(PLOTS_DIR + '/' + 'valid_bleu_' + MODEL_NAME, valid_bleu)
    np.save(PLOTS_DIR + '/' + 'test_bleu_' + MODEL_NAME, bleu_test_loss)

    mle_per_batch = reduce(lambda x,y: x+y, mle_per_batch)
    rl_per_batch = reduce(lambda x,y: x+y, rl_per_batch)
  
    np.save(PLOTS_DIR + '/' + 'mle_per_batch_' + MODEL_NAME, mle_per_batch)
    np.save(PLOTS_DIR + '/' + 'rl_per_batch_' + MODEL_NAME, rl_per_batch)


if __name__ == "__main__":
    print('testing', args.testing)
    print(f'**reward type:{args.reward_type}**')
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
    elif (args.testing == 4):
        print('Starting Training new model WITH RL(BLEU)')
        use_code_reward = False
        include_RL = True
        run_train_new_model(include_RL)          
    elif (args.testing == 5):
        print('Starting Training new model WITH RL(CODE)')
        use_code_reward = True
        include_RL = True
        run_train_new_model(include_RL, use_code_reward)      