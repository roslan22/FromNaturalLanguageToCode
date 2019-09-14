import argparse
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random
from transformer_model import evaluation
from eval.bleu_score import compute_bleu
from code_evaluator import evaluation_processor
from preproc.util import encoded_code_tokens_to_code_sl
from preproc.canonicalize import decanonicalize_code
import ast 
from torch.autograd import Variable
torch.manual_seed(1)
import enum 

"""
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2) # move right or left 

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)
"""

class RewardType(enum.Enum): 
    execution = 1
    exec_ast = 2
    exec_ast_fra = 3 # fra: future reward prediction

class Enviroment():
    def __init__(self):
        self.prev_reward = 0
        self.code_evaluator = evaluation_processor.EvaluationProcessor('enviroment_processor')

    def set_target_sentence(self, target_sentence):
        self.target_sentence = target_sentence.split()

    def set_target_sentence_batch(self, target_sentences):
        self.target_sentence = target_sentences

    def set_eos_index(self, eos_index):
        self.eos_index = eos_index

    def set_reward_type(self, reward_type):
        self.reward_type = reward_type

    def step(self, action, state, src, t, TRG_DATA, code_reward):
        new_state = torch.cat([state, torch.ones(1, 1).type_as(src.data).fill_(action.data[0])], dim=1) #state = ys     
        new_sentence = evaluation.translate2(new_state, TRG_DATA)
        #using reward shaping (Ng et al 1999) intermediate reward at each step t is imposed and denoted as r_t(y_t^hat, y)
        #if t-1 >= len(self.target_sentence):
            #t = len(self.target_sentence) - 2
        reward = self.calculate_reward(new_sentence, self.target_sentence, code_reward)         
        #next_reward = reward - self.prev_reward
        next_reward = reward
        self.prev_reward = reward
        #without prev. reward 
        #reward = self.calculate_reward(new_sentence, self.target_sentence, code_reward)
        done = True if int(action) == self.eos_index else False

        final_decoded_sentence = None
        if(done):
            final_decoded_sentence = encoded_code_tokens_to_code_sl(new_sentence.split())

        return new_state, next_reward, done, new_sentence, final_decoded_sentence

    def step_batch(self, action, state, src, t, TRG_DATA, code_reward, skip_delta):
        addition = torch.ones(src.shape[0], 1).type_as(src.data)
        for idx, val in enumerate(addition):
            addition[idx] = action[idx]

        new_state = torch.cat([state, addition], dim=1) #state = ys     
        new_sentence = evaluation.translate_batch(new_state, TRG_DATA)
        #using reward shaping (Ng et al 1999) intermediate reward at each step t is imposed and denoted as r_t(y_t^hat, y)
        #if t-1 >= len(self.target_sentence):
            #t = len(self.target_sentence) - 2
        rewards = self.calculate_reward_batch(new_sentence, self.target_sentence, code_reward, skip_delta)         
        #next_reward = reward - self.prev_reward
        next_rewards = rewards
        self.prev_reward = rewards
        #without prev. reward 
        #reward = self.calculate_reward(new_sentence, self.target_sentence, code_reward)
        done = True

        # exit if everything is zero
        for word_idx in action:
            if int(word_idx) != self.eos_index:
                done = False
                break

        final_decoded_sentence = None
        if(done):
            # final_decoded_sentence = encoded_code_tokens_to_code_sl(new_sentence.split())
            final_decoded_sentence = ''

        return new_state, next_rewards, done, new_sentence, final_decoded_sentence

    def reset(self):
        self.prev_reward = 0

    def calculate_reward(self, sentences, reference_sentences, code_reward):
        reward = 0
        rewards = []
        if(code_reward):
            for sentence in sentences:
                decoded_sentence = encoded_code_tokens_to_code_sl(sentence)
                code = decanonicalize_code(decoded_sentence, {})
                reward = self.code_evaluator.evaluate([code], single_run=True, apply_exception_fix=True)
                rewards.append(reward)
        else:
            #target_sentence = evaluation.translate2(tgt, TRG_DATA)  
            #skipping <sos> token at the beggining
            #sentences = np.array(sentence)[:, 1:].tolist()
            # skipping sos token at the beggining
            sentences = [x[1:] for x in sentences]
            reference_sentences = [[x] for x in reference_sentences]
            reward_bleu = compute_bleu(reference_sentences, sentences)
            reward = reward_bleu[0]

        return reward

    def window_stack(self, a, stepsize=1, width=3):
        return np.hstack( a[i:1+i-width or None:stepsize] for i in range(0,width))

    def calculate_reward_batch(self, sentences, reference_sentences, code_reward, skip_delta):
        reward = 0
        rewards = []
        if(code_reward):
            for sentence in sentences:
                # if lenght is one no need to evaluate code:       
                if skip_delta:
                    rewards.append(0)
                    continue

                if len(sentence[1:]) == 1:
                    rewards.append(0)
                    continue
                if(sentence[-1] == '#newline#' or sentence[-1] == '<eos>'):
                    rewards.append(-0.00000001)
                    continue
                                    
                decoded_sentence = encoded_code_tokens_to_code_sl(sentence[1:])
                #code = decanonicalize_code(decoded_sentence, {})
                #reward = self.code_evaluator.evaluate([decoded_sentence], single_run=True, apply_exception_fix=True)
                reward = self.getCodeReward(decoded_sentence)
                rewards.append(reward)
        else:
            #sentences = np.array(sentence)[:, 1:].tolist()
            # skipping sos token at the beggining
            sentences = [x[1:] for x in sentences]

            reference_sentences = [[x] for x in reference_sentences]
            for idx, sentence in enumerate(sentences):
                if len(sentence) == 0:
                    sentence = [' ']

                #if(sentence[-1] == '#newline#' or sentence[-1] == '<eos>'):
                    #rewards.append(-0.00000001)
                    #continue 

                reward_bleu = compute_bleu([reference_sentences[idx]], [sentence])
                rewards.append(reward_bleu[0])

        return rewards    

    def getCodeReward(self, decoded_sentence):
        candidates = [')', ']', '}']
        if(self.reward_type == 1):  #only execution
           reward = self.code_evaluator.evaluate([decoded_sentence], single_run=True, apply_exception_fix=False)
           return reward 

        if(self.reward_type == 2):  #execution + exception fix
           reward = self.code_evaluator.evaluate([decoded_sentence], single_run=True, apply_exception_fix=True)
           return reward 

        if(self.reward_type == 3): #exec + ast 
           reward = self.code_evaluator.evaluate([decoded_sentence], 
            single_run=True, apply_exception_fix=True, include_ast = True)
           return reward 

        # last case exec + ast + prediction fix
        try:
            #reward = self.code_evaluator.evaluate([decoded_sentence], single_run=True, apply_exception_fix=False)
            #if reward == 1:
                #return reward
            
            # try ast now
            py_ast = ast.parse(decoded_sentence)
            return 1  # was working with 1                 
            # not parsable
        except:
            for cand in candidates:
                try:
                    py_ast = ast.parse(decoded_sentence + cand)
                    return 0.1 
                except:
                    pass
        return 0

class Policy():
    def __init__(self, model):
        self.saved_log_probs = []
        self.rewards = []
        self.model = model    

#policy = Policy()
#optimizer = optim.Adam(policy.parameters(), lr=1e-2)
#eps = np.finfo(np.float32).eps.item()


def select_action_single(policy, src, t, SRC_DATA, beam_size,  state = None):
    #F-prop
    next_word_probs, next_word_indices, state = evaluation.get_single_word_probabilities(policy.model, src, 256, SRC_DATA.vocab.stoi[SRC_DATA.init_token], beam_size, t, state)
    m = Categorical(next_word_probs) # has to be word
    action = m.sample()
    next_word_indx = next_word_indices[:,int(action)]
    policy.saved_log_probs.append(m.log_prob(action))
    #building new state

    return next_word_indx, state

def select_action_batch(policy, src, t, SRC_DATA, beam_size, memory, src_mask, skip_delta, target_len, state = None):
    #F-prop
    chosen_indices = []
    next_word_probs, next_word_indices, state = evaluation.get_batch_word_probabilities(policy.model, src, 64, SRC_DATA.vocab.stoi[SRC_DATA.init_token], beam_size, t, memory, src_mask, skip_delta, state)
    normalized_prob = F.normalize(next_word_probs, p=1, dim=1)
    m = Categorical(normalized_prob) # has to be word
    action = m.sample()
    for idx, choice in enumerate(action): 
        chosen_indices.append(next_word_indices[idx, int(choice)])
    policy.saved_log_probs.append(m.log_prob(action))
    #building new state

    return chosen_indices, state

def finish_episode(policy, optimizer, eps, gamma = 0.9):
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R # gamma is discount factor
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1)        
    optimizer.step()
    
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def finish_episode_batch(policy, eps, gamma = 0.99):
    R = 0
    policy_loss = []
    returns = []
    
    # should be reward for every line
    for r in policy.rewards[::-1]:
        R = r + gamma * R # gamma is discount factor
        returns.insert(0, R)
    returns = torch.tensor(returns)
    # since we applied reward shaping to need for next step
    # applying baseline
    if returns.size()[0] > 1:
        returns = (returns - returns.mean()) / (returns.std() + eps)
    
    for log_prob, R in zip(policy.saved_log_probs, returns): 
        policy_loss.append(-log_prob * R)

    # sum losses for every sample from temp 0 to step 40
    policy_loss = torch.sum(torch.stack(policy_loss), dim = 0) / 40
    policy_loss = policy_loss.mean()
    #policy_loss = torch.cat(policy_loss)
    #policy_loss.backward(retain_graph=True)
    #torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1)        

    del policy.rewards[:]
    del policy.saved_log_probs[:]
    torch.cuda.empty_cache()

    return policy_loss

def finish_episode_batch2(policy, eps, gamma = 0.9, rl_include_baseline = True):
    policy_loss_all =[]
    policy.rewards = np.array(policy.rewards).transpose()
    baseline_avg = 0 

    for idx, sentence_reward in enumerate(policy.rewards):
        R = 0
        policy_loss = []
        returns = []
    # should be reward for every line
        for r in sentence_reward[::-1]:
            if r == -0.00000001:
                continue
            R = r + gamma * R # gamma is discount factor
            returns.insert(0, R)

        avg_value = sum(returns) / len(returns)
        baseline_avg = baseline_avg + ((avg_value - baseline_avg ) / (idx + 1))

        returns = torch.tensor(returns)
        # since we applied reward shaping to need for next step
        # applying baseline
        if rl_include_baseline == True:
            if returns.size()[0] > 1:
                #returns = (returns - returns.mean() * 0.8) / (returns.std() + eps)
                returns = returns - baseline_avg

        for log_prob, R in zip(policy.saved_log_probs, returns): 
            policy_loss.append(-log_prob[idx] * R)

        # sum losses for every sample from temp 0 to step 40
        policy_loss = torch.sum(torch.stack(policy_loss), dim = 0) / len(policy_loss) #/40 # maybe devide by 40
        policy_loss_all.append(policy_loss)

    policy_loss_all = torch.mean(torch.stack(policy_loss_all))       

    del policy.rewards
    del policy.saved_log_probs[:]
    #torch.cuda.empty_cache()

    return policy_loss_all

def finish_episode_batch_real(policy, eps, gamma = 0.99):
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R # gamma is discount factor
        returns.insert(0, R)
    returns = torch.tensor(returns)
    # since we applied reward shaping to need for next step
    if returns.size()[0] > 1:
        returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)

    policy_loss = torch.cat(policy_loss).mean()
    #policy_loss.backward(retain_graph=True)
    #torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1)        

    del policy.rewards[:]
    del policy.saved_log_probs[:]
    torch.cuda.empty_cache()

    return policy_loss

def run_reinforce(model, train_iter, SRC_DATA, TRG_DATA, code_reward = False, n_epoch=1, beam_size=1, log_interval = 10, reward_threshold = 100):
    policy = Policy(model)
    optimizer = optim.Adam(policy.model.parameters(), lr=1e-4)
    eps = np.finfo(np.float32).eps.item()
    env = Enviroment()
    running_reward = 10
    i_episode = 0
    #translator = onmt.translate.Translator()
    #for i_episode in count(1):
    state, ep_reward = 0, 0
    model.train()    
    epoch_loss = 0    
    t = 0
    for i_epoch in range(n_epoch):
        for batch in train_iter:
            src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                                else (batch.src, None)
            tgt = batch.trg
            optimizer.zero_grad()        

            # F-prop through the model.
            #outputs, attns = self.model(src, tgt, src_lengths)
            t = 0
            state = None
            ep_reward = 0
            for t in range(256):
                action, state = select_action_single(policy, src, tgt, t, SRC_DATA, TRG_DATA, beam_size, state) # predict word 
                state, reward, done, sentence = env.step(action, state, src, tgt, TRG_DATA, t, code_reward)
                policy.rewards.append(reward)
                ep_reward += reward
                t += 1
                if done:
                    break
            finish_episode(policy, optimizer, eps)
            i_episode += 1

            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

            if i_episode % log_interval == 0:
                print('Epoch {}\t Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_epoch, i_episode, ep_reward, running_reward))
            #if running_reward > reward_threshold:
                #print("Solved! Running reward is now {} and "
                 #   "the last episode runs to {} time steps!".format(running_reward, t))
                #break
            if(running_reward == 0.00):
                print("Got to episode 200")
                break
    return running_reward

def run_reinforce_on_batch(model, batch, SRC_DATA, TRG_DATA, code_reward, prediction_file, n_epoch=1, beam_size=1, log_interval = 10, reward_threshold = 100):
    policy = Policy(model)
    eps = np.finfo(np.float32).eps.item()
    env = Enviroment()
    running_reward = 10
    i_episode = 0
    t = 0
    src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                        else (batch.src, None)
    tgt = batch.trg
    tgt.requires_grad = False
    state = None
    ep_reward = 0
    all_batch_losses = []
    rewards_batch = []

    for single_src, single_trg in zip(src, tgt):
        sample_reward = 0
        env.reset()
        state = None
        target_sentence = evaluation.translate2(single_trg, TRG_DATA)  
        env.set_eos_index(TRG_DATA.vocab.stoi[TRG_DATA.eos_token])
        env.set_target_sentence(target_sentence)      

        for t in range(40): 
            action, state = select_action_single(policy, single_src, t, SRC_DATA, beam_size, state) # predict word 
            state, reward, done, sentence, final_sentence = env.step(action, state, single_src, t, TRG_DATA, code_reward)
            policy.rewards.append(reward)
            ep_reward += reward
            sample_reward += reward
            t += 1

            if done:
                break       
        
        prediction_file.write(target_sentence + '\n' + sentence + '\n' + \
            'avg reward {}'.format(sum(policy.rewards)/t) + 'rewards:' + str(policy.rewards) + '\n' + '\n')

        #adding for statistics reward per sample
        rewards_batch.append(sample_reward / t)

        all_batch_losses.append(finish_episode_batch(policy, eps))
        i_episode += 1
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    
    if i_episode % log_interval == 0:
        print('Epoch {}\t Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
            0, i_episode, ep_reward, running_reward))
    #if running_reward > reward_threshold:
        #print("Solved! Running reward is now {} and "
            #"the last episode runs to {} time steps!".format(running_reward, t))
    if(running_reward == 0.00):
        print("Got to episode 200")
    return sum(all_batch_losses) / len(all_batch_losses), rewards_batch

def run_reinforce_on_batch_real(model, batch, SRC_DATA, TRG_DATA, code_reward, prediction_file, rl_discount, rl_include_baseline, args, rl_annealing_delta = -1, n_epoch=1, beam_size=1, log_interval = 10, reward_threshold = 100,  rl_use_encoder_backprop = True):
    policy = Policy(model)
    eps = np.finfo(np.float32).eps.item()
    env = Enviroment()
    running_reward = 10
    i_episode = 0
    t = 0
    src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                        else (batch.src, None)
    trg = batch.trg
    trg.requires_grad = False
    state = None
    ep_reward = 0
    all_batch_losses = []
    rewards_batch = []
    target_len = trg.shape[1]
    env.set_reward_type(args.reward_type)

    sample_reward = 0
    env.reset()
    state = None

    trg = trg[:,1:].contiguous()
    target_sentences = evaluation.translate_batch(trg, TRG_DATA)  
    env.set_eos_index(TRG_DATA.vocab.stoi[TRG_DATA.eos_token])
    env.set_target_sentence_batch(target_sentences)      

    #src_mask = Variable(torch.ones(1, 1, src.size()[1])).to(src.device)
    src_mask = (src != 1).unsqueeze(1).unsqueeze(2)
 
    if rl_use_encoder_backprop == True:
        memory = model.encoder.forward(src, src_mask)
    else:
        with torch.no_grad():
            memory = model.encoder.forward(src, src_mask)

    for t in range(40):  
        skip_delta = t < rl_annealing_delta
        action, state = select_action_batch(policy, src, t, SRC_DATA, beam_size, memory, src_mask, skip_delta, target_len, state) # predict word 
        state, rewards, done, sentence, final_sentence = env.step_batch(action, state, src, t, TRG_DATA, code_reward, skip_delta)
        policy.rewards.append(rewards)
        ep_reward += sum(rewards)
        sample_reward += sum(rewards) / len(rewards)
        t += 1

        if done:
            break       
    #adding for statistics reward per sample
    rewards_batch.append(sample_reward)

    if args.logging_predictions == 1:
        try:
            for idx, tgt_sentence in enumerate(target_sentences):
                prediction_file.write(str(tgt_sentence) + '\n' + str(sentence[idx][1:]) + '\n' \
                    + 'rewards:' + str(np.array(policy.rewards)[:, idx]) + '\n' + '\n')
        except:
            prediction_file.write('******** ERROR OCCURED ********** ')
            print(' ******** ERROR OCCURED ********** ')
  

    loss = finish_episode_batch2(policy, eps, rl_discount, rl_include_baseline)
    i_episode += 1
    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    
    if i_episode % log_interval == 0:
        print('Epoch {}\t Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
            0, i_episode, ep_reward, running_reward))

    if(running_reward == 0.00):
        print("Got to episode 200")
    return loss , rewards_batch

if __name__ == '__main__':
    run_reinforce()
