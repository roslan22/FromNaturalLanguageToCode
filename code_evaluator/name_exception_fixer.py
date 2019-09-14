global assign_string
global assign_number 
global assign_dict
global assign_list
global assign_set
global assign_full_list

#from bs4 import BeautifulSoup
#import tkinter as tk
import os

LIST_OF_ASSIGNMENTS = ['assign_string', 'assign_number','assign_dict' , \
'assign_list', 'assign_set', 'assign_full_list', 'assign_class', 'assign_func']

class DummyClass(object):
    def call(self, *argv):
        return True

def DummyFunc(*argv):
    return True

html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>
<p class="story">...</p>
"""

class NameExceptionFixer(object):
    """ Evaluate and run fix then return True/False if succeded """    
    def IsSolvableByFix(self, snippet, error):
        copying_toggle = False
        # assignming global variable in order  to execute them
        assign_string = 'hello word'
        assign_number = 10
        assign_dict = {}
        assign_list = []
        assign_set = set()
        assign_full_list = [1,2,3,4,5,6,7,8,9,10]
        assign_class = DummyClass()
        assign_func = DummyFunc
        #assign_soup = BeautifulSoup(html_doc, 'html.parser')
        #soup = BeautifulSoup(html_doc, 'html.parser')

        if os.environ.get('DISPLAY','') != '':
            # we want to run it only where we have gui
            #root = tk.Tk()
            root = ''
        else:
            root = ''

        undefined_variable = self.ExctractUndefinedVariable(error)

        for assignment in LIST_OF_ASSIGNMENTS:     
            try:
                exec(str(undefined_variable) + "=" + assignment)
                eval(snippet)
                # getting to next line means exception solved
                return True
            # in case two variables not define in snippet we want to solve 
            # it as well, dangerous, makes process O(n^2) but should happen rarely
            except NameError as err2:
                undefined_variable2 = self.ExctractUndefinedVariable(str(err2))
                for assignment2 in LIST_OF_ASSIGNMENTS:  
                    try:
                        exec(str(undefined_variable2) + "=" + assignment2)
                        eval(snippet)                        
                        # getting to next line means exception solved
                        return True
                    except Exception as exception:
                        str(exception)
            except Exception as exception:
                str(exception)
                #print('tried to solve name {} with assigment {} but got {} exception'.format(
                #undefined_v ariable, str(assignment), str(exception)))

        return False

    """ The format of this exception is "Name \'variable Name \' not defined " """
    def ExctractUndefinedVariable(self, error):        
        copying_toggle = False
        undefined_variable = ''
        for i, character in enumerate(error):

            if copying_toggle and character is not "'":
                undefined_variable += character

            if character is "'":
                copying_toggle = not copying_toggle

        return undefined_variable
