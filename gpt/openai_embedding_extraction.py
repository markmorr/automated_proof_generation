# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:38:53 2022

@author: 16028
"""
import os
import openai
import json
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity



openai.api_key = "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"
openai.Engine.list()
openai.Completion.create(
    model="ada:ft-columbia-university-2022-04-03-22-12-33",
    prompt="Tell me what year it is.")
openai.Engine("davinci").search(
  documents=["White House", "hospital", "school"],
  query="the president"
)


ENGINE = "curie:ft-columbia-university-2022-04-21-01-49-07"
openai.Engine.list()
openai.Completion.create(
    model=ENGINE,
    prompt="Tell me what year it is.")



sample_prompt = "Prove that the set of prime numbers is infinite."
ENGINE = "text-search-curie-query-001"
embedding = openai.Embedding.create(input=sample_prompt, engine=ENGINE)['data'][0]['embedding']
len(embedding)


example_prompt_list = ['Prove that the square root of 2 is irrational.',
               'Prove that if n^2 is odd, then is odd.\nThen, prove that if n^2 is even, then n is even.',
               'Use mathematical induction to prove that $n^2 + n$ is divisible by 2 for all positive integers $n$.'
               ]

path_to_use = r'C:\Users\16028\Downloads\gpt3_data'
proof_wiki_ml_path = path_to_use + r'\ml_proofs.json'
with open(proof_wiki_ml_path) as f:
    full_data = json.load(f)
full_data = full_data['dataset']




validation_data = full_data[:8]
test_data = full_data[8:]


prompt_list = []

# engine = "text-similarity-davinci-001"
validation_embedding_list = []
for pair_dict in validation_data:
    prompt_list.append(pair_dict['prompt'])
prompt_list = [prompt.replace("\n", " ") for prompt in prompt_list]


for prompt in prompt_list:
    validation_embedding_list.append(openai.Embedding.create(input=prompt, engine=ENGINE)["data"][0]["embedding"])
    

prompt_list = []
test_embedding_list = []
for pair_dict in test_data:
    prompt_list.append(pair_dict['prompt'])
prompt_list = [prompt.replace("\n", " ") for prompt in prompt_list]


test_embedding_list = []
prompt_list = [prompt.replace("\n", " ") for prompt in prompt_list]
engine = "text-similarity-davinci-001"
for prompt in prompt_list:
    test_embedding_list.append(openai.Embedding.create(input=prompt, engine=ENGINE)["data"][0]["embedding"])
    

print(len(embedding))

np.inf

k = 3
def get_best_few_shot(distance_metric, k)
distance_matrix = euclidean_distances(test_embedding_list, validation_embedding_list) 
distance_matrix = np.array(distance_matrix)
k = 3
indices = np.argpartition(distance_matrix, k)
indices = np.array(indices)


closest_3_list = []

for i in range(indices.shape[0]):
        top_3_examples = indices[i][:k]
        
        
        
def get_best_few_shot(distance_metric, k):
    distance_matrix = distance_metric(test_embedding_list, validation_embedding_list) 
    distance_matrix = np.array(distance_matrix)
    indices = np.argpartition(distance_matrix, k)
    indices = np.array(indices)
    
    closest_3_list = []
    
    for i in range(indices.shape[0]):
            top_3_examples = indices[i][:k]
            
            closest_3_list.append(top_3_examples)
    return closest_3_list


get_best_few_shot(euclidean_distances, 3)
get_best_few_shot(cosine_similarity, 3)



#use these in Playground after selecting proper model













