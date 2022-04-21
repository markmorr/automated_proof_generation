# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:46:38 2022

@author: 16028
"""
path_to_use = r'C:\Users\16028\Downloads\gpt3_data'
    # naturalproofs_proofwiki.json'
proofwiki_json = './data/naturalproofs_proofwiki.json'
proofwiki_json = path_to_use + r'\naturalproofs_proofwiki.json'
# stacks_json = './data/naturalproofs_stacks.json'

output_json = path_to_use + r'\interim_format.json'

# cite: https://github.com/wellecks/naturalproofs/blob/master/notebooks/merge.ipynb
import json

with open(proofwiki_json) as f:
    proofwiki = json.load(f)

offset = 0
for item in proofwiki['dataset']['theorems'] + proofwiki['dataset']['definitions'] + proofwiki['dataset']['others']:
    offset = max(offset, item['id'])
offset += 1

proofwiki['dataset']['theorems']


for item in stacks['dataset']['theorems']:
    item['id'] += offset
    item['ref_ids'] = [rid + offset for rid in item['ref_ids']]
    for proof in item['proofs']:
        proof['ref_ids'] = [rid + offset for rid in proof['ref_ids']]
        
for item in stacks['dataset']['definitions']:
    item['id'] += offset
    item['ref_ids'] = [rid + offset for rid in item['ref_ids']]
    
for item in stacks['dataset']['others']:
    item['id'] += offset
    item['ref_ids'] = [rid + offset for rid in item['ref_ids']]
    
stacks['dataset']['retrieval_examples'] = [e + offset for e in stacks['dataset']['retrieval_examples']]
for split in ['train', 'valid', 'test']:
    stacks['splits'][split]['examples'] = [(tid + offset, j) for (tid, j) in stacks['splits'][split]['examples']]
    stacks['splits'][split]['ref_ids'] = [tid + offset for tid in stacks['splits'][split]['ref_ids']]

theorems = proofwiki['dataset']['theorems'] + stacks['dataset']['theorems']
definitions = proofwiki['dataset']['definitions'] + stacks['dataset']['definitions']
others = proofwiki['dataset']['others'] + stacks['dataset']['others']
retrieval_examples = proofwiki['dataset']['retrieval_examples'] + stacks['dataset']['retrieval_examples']

splits = {
    'train': {
        'examples': proofwiki['splits']['train']['examples'] + stacks['splits']['train']['examples'],
        'ref_ids': proofwiki['splits']['train']['ref_ids'] + stacks['splits']['train']['ref_ids'],
    },
    'valid': {
        'examples': proofwiki['splits']['valid']['examples'] + stacks['splits']['valid']['examples'],
        'ref_ids': proofwiki['splits']['valid']['ref_ids'] + stacks['splits']['valid']['ref_ids'],
    },
    'test': {
        'examples': proofwiki['splits']['test']['examples'] + stacks['splits']['test']['examples'],
        'ref_ids': proofwiki['splits']['test']['ref_ids'] + stacks['splits']['test']['ref_ids'],
    },
}

js = {
    'dataset': {
        'theorems': theorems,
        'definitions': definitions,
        'others': others,
        'retrieval_examples': retrieval_examples,
    },
    'splits': splits,
}


with open(output_json, 'w') as f:
    json.dump(js, f, indent=4)