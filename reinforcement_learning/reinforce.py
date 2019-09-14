import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v1')
env.seed(args.seed)
torch.manual_seed(args.seed)

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

class Policy():
    def __init__(self, model):
        self.saved_log_probs = []
        self.rewards = []
        self.model = model    

#policy = Policy()
#optimizer = optim.Adam(policy.parameters(), lr=1e-2)
#eps = np.finfo(np.float32).eps.item()


def select_action(state, policy, src, trg):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy.model(src, trg)
    m = Categorical(probs) # has to be word
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode(policy, optimizer, eps):
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)        
    optimizer.step()
    
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def run_reinforce(model, train_iterator):
    policy = Policy(model)
    optimizer = optim.Adam(policy.model.parameters(), lr=1e-2)
    eps = np.finfo(np.float32).eps.item()

    running_reward = 10
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        model.train()    
        epoch_loss = 0    
        for i, batch in enumerate(train_iterator):            
            src = batch.src
            trg = batch.trg     

            action = select_action(state, src, policy, trg) # same as predicting sentence      
            state, reward, done, _ = env.step(action)
            optimizer.zero_grad()        
            output = model(src, trg[:,:-1])                
            #output = [batch size, trg sent len - 1, output dim]
            #trg = [batch size, trg sent len]            
            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:,1:].contiguous().view(-1)                
            #output = [batch size * trg sent len - 1, output dim]
            #trg = [batch size * trg sent len - 1]        
            #loss = criterion(output, trg)        
            #loss.backward()        

            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode(policy, optimizer, eps)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break
    return running_reward

if __name__ == '__main__':
    main()
