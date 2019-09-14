import sys
import os
sys.path.append(os.path.abspath('eval'))
sys.path.append(os.path.abspath('preproc'))
import eval.conala_eval as conala_eval
import util

class ConalaEval(object):
    def __init__(self, path):
        self.path = path # can be deleted

    def calculate_bleu_score(self, hypothesis_list, target_file_name):
        reference_file_name = os.path.join(os.getcwd(), target_file_name)
        reference_list = conala_eval.get_reference_list(reference_file_name)
        #hypothesis_list = [util.encoded_code_tokens_to_code(x.split()) for x in hypothesis_list]
        hypothesis_list = [util.encoded_code_tokens_to_code_sl(x.split()) for x in hypothesis_list]

        return conala_eval.evaluate_bleu(reference_list, hypothesis_list)