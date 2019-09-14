import os
from evaluate import ConalaEval
import json 
import util

BASE_DIR = os.path.join( os.path.dirname( __file__ ), '..' )

def convert_to_competition_format(path, from_name):    
    with open(BASE_DIR + '/' + path + '/' + from_name, 'r', encoding='UTF-8') as f:
        model_results = f.readlines()
    
    # Building JSON txt format in a brute - force way
    with open(BASE_DIR + '/' + path + '/answer.txt', 'w', encoding='utf-8') as outfile:
        outfile.writelines('[')
        outfile.write('\n')
        for idx, line in enumerate(model_results):
            line_decoded = util.encoded_code_tokens_to_code_sl(line.split())
            outfile.write('\"' + str(line_decoded).replace('\n', '\\n').replace('"', '') + '\"')
            if idx != len(model_results) - 1:
                outfile.write(',')
            outfile.write('\n')
        outfile.writelines(']')
    
    with open(BASE_DIR + '/' + path + '/answer.txt', "r") as f:
         data = json.load(f)
         print(data)
    #with open(BASE_DIR + '/' + 'results/lstm-attention/answer.txt', 'w', encoding='utf-8') as 

if __name__ == "__main__":
    # **************** lstm **********************
    #convert_to_competition_format('results/lstm-attention', 'attention_test_results')
    # **************** transformers **************
    convert_to_competition_format('results/transformers', 'transformers_test_results_17.39')