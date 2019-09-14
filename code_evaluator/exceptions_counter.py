from code_evaluator.exception_types import ExceptionTypes

class ExeptionsCounter:
  def __init__(self):
    self.ex_type = ExceptionTypes()
    self.counterByName = {
        self.ex_type.OS_ERROR: 0,
        self.ex_type.VALUE_ERROR: 0,
        self.ex_type.RUNTIME_ERROR: 0,
        self.ex_type.TYPE_ERROR: 0,
        self.ex_type.NAME_ERROR: 0,
        self.ex_type.ASSERTION_ERROR: 0,
        self.ex_type.SYNTAX_ERROR: 0,
        self.ex_type.UNKNOWN_ERROR: 0
    }

  def get(self, counter_name):
    return self.counterByName[counter_name]
    
  def increment(self, counter_name):
    self.counterByName[counter_name] += 1
