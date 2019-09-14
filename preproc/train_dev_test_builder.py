import pandas as pd
import numpy as np
import sys
import pickle
from pylab import rcParams
from multiprocessing import Process
from multiprocessing.pool import ThreadPool
import random
from collections import defaultdict

import os 
BASE_DIR = os.path.join( os.path.dirname( __file__ ), '..' )

def parse_variables(intents, snippets):
    return_intents = []
    return_snippets =[]

    for intent, snippet in zip(intents, snippets):
        new_intent = []
        new_snippet = []
        var_by_name = defaultdict()
        str_by_name = defaultdict()

        for int_token in intent.split():            
            if(int_token[0] == '`' and len(int_token) > 2):
                clear_token = int_token.replace('`', '')
                var_by_name[clear_token] = '<VAR' + str(len(var_by_name)) + '>'
                new_intent.append(var_by_name[clear_token])
                continue
            if(int_token[0] == "'" and len(int_token) > 2) :
                clear_token = int_token.replace("'", '').strip()
                str_by_name[clear_token] = '<STR' + str(len(var_by_name)) + '>'
                new_intent.append(str_by_name[clear_token])
                continue
            # not string or var
            new_intent.append(int_token)

        for snp_token in snippet.split():
            clear_token = snp_token.replace("'", '').replace("`", '').strip()
            if(clear_token in var_by_name):
                new_snippet.append(var_by_name[clear_token])
                continue

            if(clear_token in str_by_name):
                new_snippet.append(str_by_name[clear_token])
                continue

            # not string or var
            new_snippet.append(snp_token)
        
        return_intents.append(' '.join(new_intent).strip() + '\n')
        return_snippets.append(' '.join(new_snippet).strip() + '\n')

    return return_intents, return_snippets

# parsing and saving training snippets **************************
with open('data/raw/conala-train.intent', 'r', encoding="UTF-8") as f:
    train_intents = f.readlines()

with open('data/raw/conala-train.snippet', 'r', encoding="UTF-8") as f:
    train_snippets = f.readlines()

train_int, train_snip = parse_variables(train_intents, train_snippets)
with open('data/raw/train-v.intent', 'w', encoding="UTF-8") as f:
    f.writelines(train_int)
with open('data/raw/train-v.snippet', 'w', encoding="UTF-8") as f:
    f.writelines(train_snip)

# parsing and saving mined snippets + eval
with open('data/raw/conala-mined.intent', 'r', encoding="UTF-8") as f:
    mined_intents = f.readlines()
with open('data/raw/conala-mined.snippet', 'r', encoding="UTF-8") as f:
    mined_snippets = f.readlines()

mined_int, mined_snip = parse_variables(mined_intents, mined_snippets)

with open('data/raw/conala-validation.intent', 'w', encoding="UTF-8") as f:
    f.writelines(mined_int[0:10_000])
with open('data/raw/conala-validation.snippet', 'w', encoding="UTF-8") as f:
    f.writelines(mined_snip[0:10_000])

with open('data/raw/conala-mined-v.intent', 'w', encoding="UTF-8") as f:
    f.writelines(mined_int[10_001:70_000])
with open('data/raw/conala-mined-v.snippet', 'w', encoding="UTF-8") as f:
    f.writelines(mined_snip[10_001:70_000])


# parsing and saving test snippets
with open('data/raw/conala-test.intent', 'r', encoding="UTF-8") as f:
    test_intents = f.readlines()
with open('data/raw/conala-test.snippet', 'r', encoding="UTF-8") as f:
    test_snippets = f.readlines()

test_int, test_snip = parse_variables(test_intents, test_snippets)

with open('data/raw/conala-test-v.intent', 'w', encoding="UTF-8") as f:
    f.writelines(test_int)
with open('data/raw/conala-test-v.snippet', 'w', encoding="UTF-8") as f:
    f.writelines(test_snip)

print('all done!')


