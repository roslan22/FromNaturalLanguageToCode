import torch
import torch.nn as nn
import reinforcement_learning.rl_reinforce as rl
import gc 
import time
from transformer_model.evaluation import epoch_time, translate2
import itertools
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

def train(model, iterator, optimizer, criterion, clip,  SRC, TRG, rl_enabled, 
    beam_size, code_reward, SAVE_DIR, MODEL_NAME_SAVE, 
    rl_discount, rl_include_baseline, rl_annealing_delta, rl_optimization_steps, args,
    prediction_log_file = None, rl_alpha = 0.05, 
    rl_use_encoder_backprop = True):  

    if(rl_enabled):
        print('***********Starting Training with RL************')
        print(f'rl_discount: {rl_discount}, baseline_included: {rl_include_baseline}, rl_alpha:{rl_alpha}, beam_size{beam_size}')

    model.train()
    
    epoch_loss = 0
    
    mle_epoch_batches_loss = []
    rl_apoch_batches_loss = []

    all_rewards = []

    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        if(rl_optimization_steps != -1 and rl_enabled == True):
            if(i % rl_optimization_steps == 0 ):
                print(f'optimization zero_grad batch {i}')                 
                optimizer.zero_grad()
        else:
            optimizer.zero_grad()
        
        output = model(src, trg[:,:-1], rl_enabled)
                
        #output = [batch size, trg sent len - 1, output dim]
        #trg = [batch size, trg sent len]
            
        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg sent len - 1, output dim]
        #trg = [batch size * trg sent len - 1]
            
        loss = criterion(output, trg)
        
        if(rl_enabled == False):
            loss.backward() 
            #total_norm = calculateNormOfParameters(model)
            #print(f'NO RL: norm before clipping: {total_norm}')       
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)        
            optimizer.step()        
            epoch_loss += loss.item()
            for single_src, single_trg in zip(batch.src, batch.trg):
                target_sentence = translate2(single_trg, TRG) 
                if args.logging_predictions == 1: 
                    prediction_log_file.write(target_sentence + 'loss:{}'.format(str(float(loss))) + '\n')
            
            if args.logging_predictions == 1: 
                prediction_log_file.flush()
        else:
            #rl.run_reinforce(model, iterator, SRC, TRG, 1)
            start_time = time.time()
            #loss.backward(retain_graph=True)
            
            loss_rl, rewards = rl.run_reinforce_on_batch_real(model, batch, SRC, TRG, 
                code_reward, prediction_log_file, rl_discount, 
                rl_include_baseline, args, rl_annealing_delta = rl_annealing_delta, 
                rl_use_encoder_backprop=rl_use_encoder_backprop, 
                beam_size = beam_size)
            
            mle_epoch_batches_loss.append(loss.item())
            rl_apoch_batches_loss.append(loss_rl.item())

            if args.logging_predictions == 1:
                try:
                    prediction_log_file.flush()
                except:
                    pass

            all_rewards.append(rewards)
            total_loss = (1 - rl_alpha) * loss + rl_alpha * loss_rl
            total_loss.backward()  

            #plot_grad_flow(model.named_parameters())

            #total_norm = calculateNormOfParameters(model)
            #print(f'norm before clipping: {total_norm}')

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)    

            if(rl_optimization_steps != -1 ):
                if((rl_optimization_steps + 1 ) % 3 == 0):
                    print(f'optimization started batch {i}')                  
                    optimizer.step()   
            else: 
                optimizer.step()

            epoch_loss += total_loss.item()            
            end_time = time.time()
            batch_mins, batch_secs = epoch_time(start_time, end_time)
            if(i % 5 == 0 and args.print_progress == 1):
                print(f'Batch: {i} | mle_loss:{loss}, rl_loss:{loss_rl}, total loss:{total_loss} | Time: {batch_mins}m {batch_secs}s ')
            del loss_rl
            del loss 
            gc.collect()
    
    # squizing list of lists 
    all_rewards = list(itertools.chain(*all_rewards))
    return epoch_loss / len(iterator), np.sum(all_rewards), mle_epoch_batches_loss, rl_apoch_batches_loss

def calculateNormOfParameters(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()