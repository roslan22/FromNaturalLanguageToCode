from code_evaluator.exceptions_counter import ExeptionsCounter
from code_evaluator.exception_types import ExceptionTypes
from code_evaluator.name_exception_fixer import NameExceptionFixer
from code_evaluator.utils import Utils
import numpy as np
import pandas as pd
import ast 

BLACK_LIST = ['sys.exit(', 'quit', 'psutil.Process', 'chdir', 
              'import tensorflow', 'os.execv', 'os.startfile', 'time.sleep', 
              'help(', 'os.path.getsize', 'sys.stdout.close(', 'eval(', 'os.kill',
              'TKAgg', 'os._exit', 'timer.st', 'os.system', 'os.chdir', 'open(', 'input(' ]

class EvaluationProcessor:
  def __init__(self, processor_name):
    self.proc_name = processor_name
    self.ex_type = ExceptionTypes()
    self.nameExceptionFixer = NameExceptionFixer()
  
  def FixByAddingImportsAndVariables(self):
    global re
    global numpy 
    global np
    global os
    global time
    global itertools
    global random
    global a 
    global df 
    global d
    global matplotlib
    global plt
    global pyplot
    global datetime
    global socket 
    global pandas 
    global sys 
    global json 
    global urllib
    global struct
    global ax
    global f
    global fig

    re = __import__('re') 
    numpy = __import__('numpy') 
    os = __import__('os') 
    time = __import__('time') 
    itertools = __import__('itertools') 
    random = __import__('random') 
    matplotlib = __import__('matplotlib') 
    plt = matplotlib.pyplot
    pyplot = matplotlib.pyplot
    datetime =  __import__('datetime')
    socket = __import__('socket')
    pandas = __import__('pandas')
    sys = __import__('sys')
    json = __import__('json')
    urllib = __import__('urllib')
    struct = __import__('struct')
    #cv2 = __import__('cv2')

    try:
      f
    except NameError:
      f, ax = plt.subplots()
      fig = f

    np = numpy
    a = ''
    try:
      df
    except NameError:
      df = pd.DataFrame([])

    d = dict()

  def isSnipetSafe(self, snippet):
    for rule in BLACK_LIST:

      if rule in snippet:
        return False
    
    return True

  def isAST(self, snippet):
    try:
      py_ast = ast.parse(snippet)
      return True
    # not parsable
    except:
      return False

  def evaluate(self, code_snippets, apply_exception_fix=False, single_run=False, include_ast = False):
    snippets_length = len(code_snippets)
    errors_by_type = {
			self.ex_type.OS_ERROR: np.empty(snippets_length, dtype=object),
			self.ex_type.VALUE_ERROR: np.empty(snippets_length, dtype=object),
			self.ex_type.RUNTIME_ERROR: np.empty(snippets_length, dtype=object),
			self.ex_type.TYPE_ERROR: np.empty(snippets_length, dtype=object),
			self.ex_type.NAME_ERROR: np.empty(snippets_length, dtype=object),
			self.ex_type.ASSERTION_ERROR: np.empty(snippets_length, dtype=object),
			self.ex_type.SYNTAX_ERROR: np.empty(snippets_length, dtype=object),
			self.ex_type.UNKNOWN_ERROR: np.empty(snippets_length, dtype=object)
	  }

    count_successfull_eval = 0
    counter = ExeptionsCounter()
    counter_try = 0 
    skipped = 0
    counter_fixed = 0
    failed_name_error = []
    is_single_run_failed = False

    if apply_exception_fix:
      self.FixByAddingImportsAndVariables()

    for snippet in code_snippets:
      try:
        counter_try += 1    
        is_AST = False

        if include_ast == True:  
          is_AST = self.isAST(snippet)
        
        # return reward when is AST
        if(single_run == True and is_AST == True):
          #print(snippet + ' is ' + 'AST')
          return 0.1

        if self.isSnipetSafe(snippet):
          eval(snippet)
          count_successfull_eval += 1        
        else:
          #print('skipped snippet {}'.format(snippet))
          skipped +=1
          is_single_run_failed = True


      except OSError as err:
          errors_by_type[self.ex_type.OS_ERROR][counter.get(self.ex_type.OS_ERROR)] = \
				  {'error' : str(err), 'snippet' : snippet }
          counter.increment(self.ex_type.OS_ERROR)
          is_single_run_failed = True
      except ValueError as err:        
        errors_by_type[self.ex_type.VALUE_ERROR][counter.get(self.ex_type.VALUE_ERROR)] = \
				{'error' : str(err), 'snippet' : snippet }
        counter.increment(self.ex_type.VALUE_ERROR)   
        is_single_run_failed = True     
      except RuntimeError as err:
        errors_by_type[self.ex_type.RUNTIME_ERROR][counter.get(self.ex_type.RUNTIME_ERROR)] = \
				{'error' : str(err), 'snippet' : snippet }
        counter.increment(self.ex_type.RUNTIME_ERROR)  
        is_single_run_failed = True      
      except TypeError as err:
        errors_by_type[self.ex_type.TYPE_ERROR][counter.get(self.ex_type.TYPE_ERROR)] = \
				{'error' : str(err), 'snippet' : snippet }
        counter.increment(self.ex_type.TYPE_ERROR)    
        is_single_run_failed = True    
      except NameError as err:
        isFixed = False
        is_single_run_failed = True
        if(apply_exception_fix):
          isFixed = self.nameExceptionFixer.IsSolvableByFix(snippet, str(err))  

        if isFixed:
          # no need to count as bad exception
          count_successfull_eval += 1
          counter_fixed += 1
          is_single_run_failed = False

        if not isFixed:
          failed_name_error.append({'snippet' : snippet, 'error' : err})
          errors_by_type[self.ex_type.NAME_ERROR][counter.get(self.ex_type.NAME_ERROR)] = \
          {'error' : str(err), 'snippet' : snippet }
          counter.increment(self.ex_type.NAME_ERROR)        
      except AssertionError as err:
        errors_by_type[self.ex_type.ASSERTION_ERROR][counter.get(self.ex_type.ASSERTION_ERROR)] = \
				{'error' : str(err), 'snippet' : snippet }
        counter.increment(self.ex_type.ASSERTION_ERROR)
        is_single_run_failed = True
      except SyntaxError as err:
        errors_by_type[self.ex_type.SYNTAX_ERROR][counter.get(self.ex_type.SYNTAX_ERROR)] = \
				{'error' : str(err), 'snippet' : snippet }
        counter.increment(self.ex_type.SYNTAX_ERROR)    
        is_single_run_failed = True        
      except Exception as exception:
        errors_by_type[self.ex_type.UNKNOWN_ERROR][counter.get(self.ex_type.UNKNOWN_ERROR)] = \
				{'error' : str(exception), 'snippet' : snippet }
        counter.increment(self.ex_type.UNKNOWN_ERROR)
        is_single_run_failed = True
      except:
        print('some uncategorized exception occured, snippet number {} snippert {}'.format(counter_try, snippet))
        is_single_run_failed = True

      #exit on single run and return rewards
      if(single_run):
        if(is_single_run_failed):
          if(is_AST):
            return 0.1
          return 0
        else:
          return 1

      if counter_try % 50 == 0: 
        print('Processing snippet number {}'.format(counter_try))

      #if(counter_try > 140):
          #print('Current Snippet {}'.format(snippet))

    Utils.save_obj('results/', errors_by_type, self.proc_name + '_errors_by_type')
    Utils.save_obj('results/', counter, self.proc_name + '_counter_class') 
    Utils.save_obj('results/', failed_name_error, self.proc_name + '_failed_name_error')
    
    print('evaluated succesfully {} out of {} code snippets. Skipped {} snippets. \n Fixed {} snippets'.format(count_successfull_eval, counter_try, skipped, counter_fixed))
    return (errors_by_type, counter)

 