import os
from evaluate import ConalaEval

BASE_DIR = os.path.join( os.path.dirname( __file__ ), '..' )

def test_bleu_score_should_be_zero():
    evaluator = ConalaEval(BASE_DIR)
    with open(BASE_DIR + '/' + 'data_new/gold/test_conala2k.snippet', 'r', encoding='UTF-8') as f:
        real_snippets = f.readlines()

    # code is not exact since in 23 test samples transition from snipet.json to snipet not corret 
    # in some cases because of '(' character
    # maximum score is 0.981371235
    #f_reference = open('../data_new/origin/conala-test.json')
    #a = parse_file_json(f_reference)
    #a = [[l] for l in a]
    bleu = evaluator.calculate_bleu_score(real_snippets, 'data_new' + os.sep +'origin' + os.sep + 'conala-test.json')
    assert bleu[0] > 0.981371235, "Score of same test sets Should be > 0.981"

if __name__ == "__main__":
    test_bleu_score_should_be_zero()
    print("Everything passed")