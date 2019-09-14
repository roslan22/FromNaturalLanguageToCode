import os
from evaluate import ConalaEval

BASE_DIR = os.path.join( os.path.dirname( __file__ ), '..' )

def evaluate_model_results(file_name):
    evaluator = ConalaEval(BASE_DIR)
    with open(BASE_DIR + '/' + 'results/' + file_name, 'r', encoding='UTF-8') as f:
        model_output = f.readlines()

    # code is not exact since in 23 test samples transition from snipet.json to snipet not corret 
    # in some cases because of '(' character
    # maximum score is 0.981371235
    bleu = evaluator.calculate_bleu_score(model_output, 'data_new/origin/conala-test.json')
    print(f'{file_name}: BLEU score is : {round(bleu[0]*100, 3)}') 

if __name__ == "__main__":
    evaluate_model_results('transformers/pred.txt')

    # ******* LSTM *******
    #evaluate_model_results('lstm-attention/attention_test_results')
    # ******* Transformers ******
    #evaluate_model_results('transformers/transformers_test_results_16.61')

    print("Finished")