# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 23:46:06 2022

@author: 16028
"""

EXTRACT_PATH = r'C:\Users\16028\Downloads\gpt3\pattern_recognition_extract.pdf'
#!pip install pymudf
import os
import fitz
print(fitz.__doc__)
doc1 = fitz.open(EXTRACT_PATH)     # or fitz.Document(filename)

import tika
from tika import parser
parsed = parser.from_file(EXTRACT_PATH)
print(parsed["metadata"])
print(parsed["content"][:100]) #just look at the beginning of the text for sanity checks
doc2 = parsed["content"]

#COMPARE RESULTS FROM PDF1, PDF2S
#NEITHER WORKS GREAT ON THE TEXTBOOK PDF
#tika is missing a java runtime??

string_test = doc1[:100]
string = doc1.replace('\n', '$$$')
sentanceList = string.split('.')
string = '.\n'.join(sentanceList)
print(doc1)
print(string)


