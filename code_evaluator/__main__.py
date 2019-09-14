from evaluation_processor import EvaluationProcessor
import pandas as pd
import pyautogui
from threading import Timer


from data_providers import DataProvider
from evaluation_processor import EvaluationProcessor
from utils import Utils
from exception_types import ExceptionTypes
ex_type = ExceptionTypes()
data_provider = DataProvider()

def enterClicked():
    pyautogui.press('enter')

def run_conala_mined():
    conala_mined = pd.read_json('conala-corpus/conala-mined.jsonl', lines=True)

    mined_snippets = conala_mined['snippet'].values[17350:20000]

    evalProcessor = EvaluationProcessor('mined_10k')

    t = Timer(5.0, enterClicked)
    t.start()  # after 30 seconds, "hello, world" will be printed



    errors_by_type, counter = evalProcessor.evaluate(mined_snippets, True)
    print(counter)

def run_conala_test():
    conala_train = pd.DataFrame(data_provider.load_data('train'))
    snippets = conala_train['snippet'].values

    errors_by_type, counter = EvaluationProcessor('evaluation_with_imports').evaluate(snippets, True)
    print(counter)

run_conala_mined()