# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 13:41:01 2022

@author: 16028
"""

import os
import openai



openai.api_key = "MY_API_KEY_DONT_PUSH_TO_GITHUB"
openai.Engine.list()
openai.Completion.create(
    model="ada:ft-columbia-university-2022-04-03-22-12-33",
    prompt="Tell me what year it is.")
openai.Engine("davinci").search(
  documents=["White House", "hospital", "school"],
  query="the president"
)



#citing Caz Czworkowski on writing this class to use the GPT AI, https://www.youtube.com/watch?v=_N1dEamEqpg
class GPT:
    
    def __init__(self, engine='ada',
                 temperature=.5,
                 max_tokens=100,
                 frequency_penalt = .5)
        self.examples = []
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        
    def add_example(self, ex):
        self.examples.append(ex)
        
    def get_prime_text(self, ex):
        return '\n'.join(self.examples) + '\n'
    
 
prompt1 = {"prompt": "Prove that the mean is an unbiased estimator for the Gaussian distribution", 
 "completion": "Using the Central Limit Theorem, we see that the sample mean tends towards the true mean with probability 1."}

classifier = GPT.add_example(prompt1)




