# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 23:20:00 2022

@author: 16028
"""

import pandas as pd
import numpy as np
import os

import json

mypath = r'C:\Users\16028\Downloads\gpt3_my_local\a_new_approach'
if os.getcwd() != mypath:
    os.chdir(mypath)
  
tweets = []
i = 0
for line in open('25problems.json', 'r', encoding='utf-8'):
    i += 1
    print(i)
    print(line)
    try:
        tweets.append(json.loads(line, strict=False))
    except:
        i += 1
    i += 1




with open('example_json.json', encoding='utf-8') as f:
    proofwiki = json.load(f)


tweets = []
i = 0
for line in open('example_json.json', 'r', encoding='utf-8'):
    print(i)
    tweets.append(json.loads(line))
    

filename = "lean_test_text.txt" 

mypath = r'C:\Users\16028\Downloads\gpt3_my_local\a_new_approach'
if os.getcwd() != mypath:
    os.chdir(mypath)


with open(filename, "r", encoding="utf8") as file:
    full_text = file.read()


mystr = full_text.split("theorem")

my_string = 'theorem.'
s = 'theorem'


import re
pat = re.compile('\b(?=\w*[mathd])\w+\b')

a = re.search(pat, full_text)




def buildColumn(x, regex_list ): #full_RegEx
    #https://stackoverflow.com/questions/12182744/python-pandas-apply-a-function-with-arguments-to-a-series
    # regex_list = [RegEx_t_num_6_digit]
    # regex_list = [RegEx_10_any]
    building_keyword_list = []
    for regex in regex_list:
        m = regex.findall(x)
        building_keyword_list = building_keyword_list + m
    return building_keyword_list

ag = buildColumn(full_text, [pat])


    

m = re.findall('(?<=theorem)(.*)', full_text)
print(m)


mathd_list = []
for st in m:
    if 'mathd' in st:
        mathd_list.append(st)
        
aime_list = []
for st in m:
    if 'aime' in st:
        aime_list.append(st)  
        
imo_list = []
for st in m:
    if 'imo' in st:
        imo_list.append(st)  
        
amc_list = []
for st in m:
    if 'amc' in st:
        amc_list.append(st)  
        



mylist_pair = []
for name in mathd_list:
    my_file_name = name.split('_')
    subj = my_file_name[1]
    id_num = my_file_name[2]
    mylist_pair.append((subj, id_num))
    

if os.path.isfile(user_input + os.sep + fname):
    # Full path
    f = open(user_input + os.sep + fname, 'r')

    if searchstring in f.read():
        print('found string in file %s' % fname)
    else:
        print('string not found')
    f.close()
   
