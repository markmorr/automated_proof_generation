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


mypath = r'C:\Users\16028\Downloads\gpt3_my_local\a_new_approach'
if os.getcwd() != mypath:
    os.chdir(mypath)



# openai.api_key = ""
openai.api_key = "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"
openai.Engine.list()
openai.Completion.create(
    model="ada:ft-columbia-university-2022-04-03-22-12-33",
    prompt="Tell me what year it is.")
openai.Engine("davinci").search(
  documents=["White House", "hospital", "school"],
  query="the president"
)





sample_prompt = "Prove that the set of prime numbers is infinite."
ENGINE = "text-search-curie-query-001"
embedding = openai.Embedding.create(input=sample_prompt, engine=ENGINE)['data'][0]['embedding']
len(embedding)


example_prompt_list = ['Prove that the square root of 2 is irrational.',
               'Prove that if n^2 is odd, then is odd.\nThen, prove that if n^2 is even, then n is even.',
               'Use mathematical induction to prove that $n^2 + n$ is divisible by 2 for all positive integers $n$.'
               ]

path_to_use = r'C:\Users\16028\Downloads\gpt3_my_local\gpt3_data'
proof_wiki_ml_path = path_to_use + r'\ml_proofs.json'
proof_wiki_ml_path = "25problems.json"
with open(proof_wiki_ml_path) as f:
    full_data = json.load(f)
full_data = full_data['dataset']


tweets = []
i = 0
for line in open('25problems.json', 'r', encoding='utf-8'):
    print(i)
    print(line)
    try:
        tweets.append(json.loads(line, strict=False))
    except:
        i += 1
    i += 1
    
i = 0
for line in open('22proofs.json', 'r', encoding='utf-8'):
    print(i)
    print(line)
    try:
        tweets.append(json.loads(line, strict=False))
    except:
        i += 1
    i += 1
    

full_data = tweets.copy()



i = 0
new_tweets = []
for line in open('np_reformat_4_25.json', 'r', encoding='utf-8'):
    print(i)
    print(line)
    try:
        new_tweets.append(json.loads(line, strict=False))
    except:
        i += 1
    i += 1
    

full_data = tweets.copy()

full



a = backup_proof_data[4]['completion'].split('begin')
        

translation_data = full_data.copy()
backup_proof_data = full_data.copy()




    
for i, bp in enumerate(backup_proof_data):
    try: 
        bp['completion'] = bp["completion"].split('begin')[1]   
    except:
        print(i)
        
 for tp in translation_data:
     tp['completion'] = tp["completion"].split('begin')[0]
        
    
    
for i in range(len(backup_proof_data)):
    print(i)
    backup_proof_data[i]["completion"].split('begin')[1] #get the proof only
full_data = translation_data.copy()


#use 80% as support examples, .25 as query set
import random
all_indices = list(range(len(full_data)))
random.Random(9).shuffle(all_indices)

split_point = int(len(full_data)*.80)
split_point
validation_data = [full_data[i] for i in all_indices[:split_point]]
test_data = [full_data[i] for i in all_indices[split_point:]]

test_data = new_tweets

all_indices[:split_point]
all_indices[split_point:]

new_proofs = 

















tempy = .2
tempy_str = int(tempy * 10)
k = 3
mymetric = euclidean_distances
# def getResults(validation_data, test_data, k,tempy, distance_metric):

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
# engine = "text-similarity-davinci-001"
engine = "code-search-ada-text-001"
for prompt in prompt_list:
    test_embedding_list.append(openai.Embedding.create(input=prompt, engine=ENGINE)["data"][0]["embedding"])
    

def get_best_few_shot(distance_metric, k, difference_measure=True):
    distance_matrix = distance_metric(test_embedding_list, validation_embedding_list) 
    distance_matrix = np.array(distance_matrix)
    if difference_measure == True:
        indices = np.argpartition(distance_matrix, k)
    else:
        indices = np.argpartition(distance_matrix, -k)
    indices = np.array(indices)
    
    closest_K_list = []
    
    for i in range(indices.shape[0]):
        if difference_measure==True:
            top_K_examples = indices[i][:k]
        else:
            top_K_examples = indices[i][-k:]
            
        closest_K_list.append(top_K_examples)
    return closest_K_list



# few_shot_indices = get_best_few_shot(cosine_similarity, k, False)
few_shot_indices = get_best_few_shot(mymetric, k, True)











training_prompt_list = []
response_list = []
for i in range(len(test_data)):
    train_prompt = "Given the mathematical statement, translate it into the mathematical programming language Lean."
    few_shot_examples = [validation_data[j] for j in few_shot_indices[i]]
    for example in few_shot_examples:
        train_prompt += "\n" + "###" + "\n" + 'prompt: ' + example['prompt'] + "\n" + "completion: " + example["completion"] + '\n end'
    # train_prompt +=  "\n" + "###" + '\n' + 'prompt: ' + test_data[i]['prompt'] + "\n" + "completion: "
    test_question = "\n" + 'prompt: ' + test_data[i]['prompt'] + "\n" + "completion: "
    final_prompt = train_prompt + test_question
    training_prompt_list.append(final_prompt)
    response = openai.Completion.create(
      engine="code-davinci-002",
      prompt= final_prompt,
      temperature=0.1,
      max_tokens=100,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop= "end"
    )
    response = response['choices'][0]['text']
    response_list.append(response)


results_list = list(zip(training_prompt_list, response_list))

#euclidean, cosine, k =3,5,7, temperature = .2, .1
print(few_shot_indices)
print(results_list)

mytuples_list = []
for i in range(len(test_data)):
    proof = test_data[i]['prompt']
    lean_translation = test_data[i]['completion']
    lean_proof = results_list[i][1] #the second on  is the completion
    mytuples_list.append((proof, lean_translation, lean_proof))
    


myfile = open('10_test_translation_only_k_is_' + str(k) + '_temp_is_p' + str(tempy_str) + '_euclidean.txt', 'w', encoding="utf-8")
for t in mytuples_list:
    line = ' \n break \n '.join(x for x in t)
    print(line)
    # line = line.encode("utf-8")
    myfile.write(line)
    # myfile.write(line.encode("utf-8") + '\n')
    myfile.write('\n')
    myfile.write('new_problem')
    myfile.write('\n')
myfile.close()

# return

######################################################################################

mytuples_list
proof_dict_list = []
for i in range(len(mytuples_list)):
    temp_dict = dict()
    temp_dict['prompt'] = mytuples_list[i][1]
    temp_dict['completion'] = backup_proof_data[i]['completion']
    proof_dict_list.append(temp_dict)
    
# len(mytuples_list)
# len(backup_proof_data)
    
second_split_point = int(len(proof_dict_list) * .5)
temp_few_shot = proof_dict_list[:second_split_point]
temp_proofs = proof_dict_list[second_split_point:]
 

training_prompt_list = []
response_list = []   
for i in range(len(temp_proofs)):
    train_prompt = "Given the mathematical statement in Lean, write a proof of its correctness."
    few_shot_examples = temp_few_shot.copy()
    for example in few_shot_examples:
        train_prompt += "\n" + "###" + "\n" + 'prompt: ' + example['prompt'] + "completion: " + example["completion"] + '\n end'
    # train_prompt +=  "\n" + "###" + '\n' + 'prompt: ' + test_data[i]['prompt'] + "\n" + "completion: "
    test_question = "\n" + 'prompt: ' + temp_proofs[i]['prompt'] + "\n" + "completion: "
    final_prompt = train_prompt + test_question
    training_prompt_list.append(final_prompt)
    response = openai.Completion.create(
      engine="code-davinci-002",
      prompt= final_prompt,
      temperature=tempy,
      max_tokens=100,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop= "end"
    )
    response = response['choices'][0]['text']
    response_list.append(response)

my_proof_tuples_list = []
for i in range(len(temp_proofs)):
    proof = temp_proofs[i]['prompt']
    lean_translation = temp_proofs[i]['completion'] #the actual answer
    lean_proof = response_list[i] #the second one is the completion
    my_proof_tuples_list.append((proof, lean_translation, lean_proof))
response_list

    
myfile = open('5_proof__k_is_' + str(k) + '_temp_is_p' + str(tempy_str) + '_euclidean.txt', 'w', encoding="utf-8")
for t in my_proof_tuples_list:
    line = ' \n break \n '.join(x for x in t)
    print(line)
    # line = line.encode("utf-8")
    myfile.write(line)
    # myfile.write(line.encode("utf-8") + '\n')
    myfile.write('\n')
    myfile.write('new_problem')
    myfile.write('\n')
myfile.close()

# return




    
# train_prompt = "Given the mathematical theorem statement, translate it into Lean."
train_prompt = "Given the mathematical statement, write a proof of its correctness."

for example in few_shot_examples:
    train_prompt += "\n" + "###" + "\n" + 'prompt: ' + example['prompt'] + "\n" + "completion: " + example["completion"]
train_prompt +=  "\n" + 'prompt: ' + test_data[0]['prompt'] + "\n" + "completion: "
model_output = call_engine(train_prompt, problem_data["problem"], engine=engine)


response = openai.Completion.create(
  engine="code-davinci-002",
  prompt= train_prompt,
  temperature=0.3,
  max_tokens=120,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)
response = response['choices'][0]['text']


test_data[0]['completion']










