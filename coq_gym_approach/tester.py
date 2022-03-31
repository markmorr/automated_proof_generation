# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:54:48 2022

@author: 16028
"""

import pandas as pd
import numpy as np


def main(args):
    if args.model == 'myguy':
        print('hi')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', required=True,
                        choices=["dense", "RNN", "extension1", "extension2"],
                        help='The name of the model to train and evaluate.')
    args = parser.parse_args()
    main(args)
    
# Import the library
import argparse
# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--name', type=str, required=True)
# Parse the argument
args = parser.parse_args()
# Print "Hello" + the user input argument
print('Hello,', args.name)