import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os 
import itertools
from code_evaluator import evaluation_processor
import ast 
import copy

from eval.evaluate import ConalaEval
from eval.bleu_score import compute_bleu
import torchtext
from preproc.util import encoded_code_tokens_to_code_sl2, encoded_code_tokens_to_code_sl

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg[:,:-1], False)
            
            #output = [batch size, trg sent len - 1, output dim]
            #trg = [batch size, trg sent len]
            
            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg sent len - 1, output dim]
            #trg = [batch size * trg sent len - 1]
            
            loss = criterion(output, trg)

            epoch_loss += loss.item()


    return epoch_loss / len(iterator)

def translate(output, TRG):
    out = F.softmax(output, dim=1)
    _, indices = out.max(1)
    result = []

    # skipping eos 
    for idx in indices:
        if TRG.vocab.itos[idx] == TRG.eos_token:
            break
        result.append(TRG.vocab.itos[idx])
        
    return ' '.join(result)

def translate2(word_indices, TRG):
    result = []
    # skipping bos and eos
    word_indices = word_indices.squeeze(0)
    for idx in word_indices[1:]:
        if TRG.vocab.itos[int(idx)] == TRG.eos_token:
            break
        result.append(TRG.vocab.itos[int(idx)])
        
    return ' '.join(result)

def translate_batch(word_indices, TRG, skip_first = False):
    result = []

    if (skip_first):
        word_indices = word_indices[:,1:]

    for row in word_indices:
        row_translation = []
        for idx in row:
            if TRG.vocab.itos[int(idx)] != TRG.eos_token:
                row_translation.append(TRG.vocab.itos[int(idx)])
            else:
                row_translation.append(TRG.vocab.itos[int(idx)])
                break
        result.append(row_translation)
        
    return result

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def greedy_decode(model, src, max_len, start_symbol, TRG):
    src_mask = Variable(torch.ones(1, 1, src.size()[1])).to(src.device)
    memory = model.encoder.forward(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    #trg, src, trg_mask, src_mask
    out_words_matrix = []
    for i in range(max_len-1):
        target = Variable(ys)
        target_msk = Variable(subsequent_mask(ys.size(1)).type_as(src.data))
        out = model.decoder.forward(target, memory,target_msk, src_mask)
        out_words_matrix.append(out)
        
        prob = F.softmax(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        # if eos was predicted no need to continue
        if(TRG.vocab.itos[int(next_word)] == TRG.eos_token):
            break
            
    return ys

def get_single_word_probabilities(model, src, max_len, start_symbol, beam_size, index, ys = None):
    #src_mask = Variable(torch.ones(1, 1, src.size()[1]))
    # changing dimentions only for single src case 
    src = src[None,:]
    
    src_mask = Variable(torch.ones(1, 1, src.size()[1])).to(src.device)
    
    with torch.no_grad():
        memory = model.encoder.forward(src, src_mask)

    if ys is None:
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)

    #trg, src, trg_mask, src_mask    
    if(index == max_len):
        raise "cant be more than max size "

    target = Variable(ys)
    target_msk = Variable(subsequent_mask(ys.size(1)).type_as(src.data))
    out = model.decoder.forward(target, memory,target_msk, src_mask)
    prob = F.softmax(out[:, -1]) 
    next_word_probs, next_word_indices = torch.topk(prob, beam_size, dim = 1)
        #must concatenate new word to y1..y_t -> ys
    return next_word_probs, next_word_indices, ys

def get_batch_word_probabilities(model, src, max_len, start_symbol, beam_size, index, memory, src_mask, skip_delta, ys = None):
    #src_mask = Variable(torch.ones(1, 1, src.size()[1]))
    # changing dimentions only for single src case 
    #src = src[None,:]

    if ys is None:
        ys = torch.ones(src.size()[0], 1).fill_(start_symbol).type_as(src.data)

    #trg, src, trg_mask, src_mask    
    if(index == max_len):
        raise "cant be more than max size "

    target = Variable(ys)
    target_msk = Variable(subsequent_mask(ys.size(1)).type_as(src.data))
    out = model.decoder.forward(target, memory, target_msk, src_mask)
    prob = F.softmax(out[:, -1])
    if skip_delta:
        beam_size = 1

    next_word_probs, next_word_indices = torch.topk(prob, beam_size, dim = 1)
        #must concatenate new word to y1..y_t -> ys
    return next_word_probs, next_word_indices, ys

def calculate_bleu_score(model, target_data, SRC, TRG, base_dir):
    input_pad = SRC.vocab.stoi['<pad>']
    bleu_eval = ConalaEval(base_dir)
    hypothesis_list = []
    model.eval()
    
    with torch.no_grad():        
        source_list = []
        for data in target_data:
            src = torch.LongTensor([SRC.vocab.stoi[x] for x in data.src]).unsqueeze(0).to(model.device)
            source_list.append(data.src)
            
            #trg = torch.LongTensor([TRG.vocab.stoi[x] for x in data.trg]).unsqueeze(0)
            output = greedy_decode(model, src, 64, SRC.vocab.stoi[SRC.init_token], TRG)
            #pred = translate(output[0].squeeze(), TRG)
            pred = translate2(output, TRG)
            hypothesis_list.append(pred)

    bleu_score = bleu_eval.calculate_bleu_score(hypothesis_list, 'data_new' + os.sep + 'origin' + os.sep + 'conala-test.json')
    return round(bleu_score[0] * 100, 2)

def calculate_bleu_score_batch(model, data_iterator, SRC, TRG, base_dir, log_result = False):
    input_pad = SRC.vocab.stoi['<pad>']
    hypothesis_list = []
    references_list = []
    model.eval()
    
    with torch.no_grad():        
        for i, batch in enumerate(data_iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg[:,:-1], False)
            
            #output = [batch size, trg sent len - 1, output dim]
            #trg = [batch size, trg sent len]
            
            #output = output.contiguous().view(-1, output.shape[-1])
            #trg = trg[:,1:].contiguous().view(-1)

            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:,1:].contiguous()


            references = translate_batch(trg, TRG)
            references_list.append(references)

            preds_prob = F.softmax(output, dim=1)
            _, word_indices = torch.max(preds_prob, dim = 1)
            #reshape for translation
            word_indices = word_indices.view(-1, trg.shape[1])
            predictions = translate_batch(word_indices, TRG)
            hypothesis_list.append(predictions)


        hypothesis_list = list(itertools.chain.from_iterable(hypothesis_list))
        references_list = list(itertools.chain.from_iterable(references_list))

        hypothesis_list = [encoded_code_tokens_to_code_sl2(x) for x in hypothesis_list]
        references_list = [[encoded_code_tokens_to_code_sl2(x)] for x in references_list]

        bleu_score =  compute_bleu(references_list, hypothesis_list, smooth=False)

        if log_result == True:
            with open('results/final_bleu_test_res.txt', 'w', encoding='utf-8') as outfile:
                for snippet in hypothesis_list:
                    outfile.writelines(' '.join(snippet) + '\n')

    return bleu_score[0]

def calculate_bleu_score_batch_real(model, data_iterator, SRC, TRG, base_dir, execute_snippets = False, max_n_gram = 4, log_result = False):
    input_pad = SRC.vocab.stoi['<pad>']
    hypothesis_list = [] 
    references_list = []
    #just in case copying model

    model = copy.deepcopy(model)

    model.eval()
    statistics = {'executed': 0,
                'executed_with_fix': 0,
                'AST': 0, 
                'total': 0 }

    with torch.no_grad():        
        for i, batch in enumerate(data_iterator):
            ys = None
            max_len = 40
            chosen_indices_batch = []
            src = batch.src
            trg = batch.trg

            #src_mask = Variable(torch.ones(1, 1, src.size()[1])).to(src.device)  
            src_mask = (src != 1).unsqueeze(1).unsqueeze(2).to(src.device) 
          
            with torch.no_grad():
                memory = model.encoder.forward(src, src_mask)

            for t in range(max_len):
                next_word_probs, next_word_indices, ys = get_batch_word_probabilities(
                    model, src, max_len, SRC.vocab.stoi[SRC.init_token], 1, t, memory, src_mask, False, ys)
                
                addition = torch.ones(src.shape[0], 1).type_as(src.data)
                for idx, val in enumerate(next_word_indices):
                    addition[idx] = val

                ys = torch.cat([ys, addition], dim=1)
                # check if we need those 2 next steps!!!
                #preds_prob = F.softmax(next_word_probs, dim=1) # check if it returns batch of probs
                #_, word_indices = torch.max(preds_prob, dim = 1)

            references = translate_batch(trg, TRG, skip_first=True)
            references_list.append(references)

            #reshape for translation
            #word_indices = word_indices.view(-1, trg.shape[1])
            predictions = translate_batch(ys, TRG, skip_first=True)
            hypothesis_list.append(predictions)

        hypothesis_list = list(itertools.chain.from_iterable(hypothesis_list))
        references_list = list(itertools.chain.from_iterable(references_list))

        hypothesis_list = [x for x in hypothesis_list]
        references_list = [[x] for x in references_list]
        
        bleu_score =  compute_bleu(references_list, hypothesis_list, max_order = max_n_gram, smooth=False)

        if execute_snippets:
            statistics = execute_results(hypothesis_list, statistics)

        if log_result == True:
            with open('results/final_bleu_test_res.txt', 'w', encoding='utf-8') as outfile:
                for snippet in hypothesis_list:
                    outfile.writelines(''.join(snippet).replace('\n','#NEWLINE#') + '\n')

    return bleu_score[0], statistics    
    #bleu_score = bleu_eval.calculate_bleu_score(hypothesis_list, 'data_new' + os.sep + 'origin' + os.sep + 'conala-test.json')
    #return round(bleu_score[0] * 100, 2)

def execute_results(predicted_snippets, statistics):
    code_evaluator = evaluation_processor.EvaluationProcessor('enviroment_p2')

    for snippet in predicted_snippets:
        decoded_snippet = encoded_code_tokens_to_code_sl(snippet[:-2])
        # try only to execute predicted snippet
        reward = code_evaluator.evaluate([decoded_snippet], apply_exception_fix=False, single_run=True, include_ast = False)
        if reward == 1:
            statistics['executed'] += 1
        
        reward_with_fix = code_evaluator.evaluate([decoded_snippet], apply_exception_fix=True, single_run=True, include_ast = False)
        if reward_with_fix == 1:
            statistics['executed_with_fix'] += 1
        
        try:           
            # try ast now
            py_ast = ast.parse(decoded_snippet)
            statistics['AST'] += 1            
        except:
            pass

        statistics['total'] += 1

    return statistics

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

