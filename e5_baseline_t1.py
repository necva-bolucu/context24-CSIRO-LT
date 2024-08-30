import os
import json
import find_citing_contexts as fcc

from rank_bm25 import BM25Okapi
import nltk
# nltk.download('punkt')
from nltk import word_tokenize
from faiss_indexing import E5MistralVectoriser
import numpy as np
rootdir = '../context24-main/figures-tables'

# def add_parents(predictions, label_pool):
#     candidates=[]
#     for pred in predictions:
#         possible_parent_name=pred[:-1]
#         possible_parent_name=possible_parent_name.replace(".", "").replace("-", "")
#         if possible_parent_name in label_pool:
#             print('Adding ' + possible_parent_name + ' for ' + pred) 
#             candidates.append(possible_parent_name)
#     return list(set(candidates))
    
docs2elements={}
cap_count=0

for _, dirs, _ in os.walk(rootdir):
    for d in dirs:
        path = os.path.join(rootdir, d)
        print (path)
        for _, _, files in os.walk(path):
            docs2elements[d]=[f.replace(".png", "") for f in files if 'CAPTION' not in f]
            cap_count+=len([f for f in files if 'CAPTION' in f])
            # for f in files:
            #     print ('    '+ f)



with open('t1_train.json') as f:
    train = json.load(f)


with open('t1_dev.json') as f:
    dev = json.load(f)

run_new_only=True


vectoriser=E5MistralVectoriser()
if not run_new_only:
    # embedding=vectoriser.vectorise(['doc 1', 'doc 2', 'doc3'], batch_size=8)
    dev_preds_e5={}
    # q_vect = vectoriser.encode_query('element 2')
    for claim in dev:
        print(claim['id'])
        citing_contexts = fcc.get_citing_text(claim['citekey'], chunksize=100)
        citing_vectors={}
        for e in citing_contexts:
            citing_vectors[e] = vectoriser.vectorise([citing_contexts[e]])[0]
        q_vect = vectoriser.encode_query(claim['claim'], task_instr='Given a claim, retrieve descriptions of figures and tables that support the claim')[0]
        result = [(e, np.dot(citing_vectors[e], q_vect)) for e in citing_vectors]
        result.sort(key=lambda a: a[1], reverse=True)
        dev_preds_e5[claim['id']] = [r[0] for r in result]
        
        
    dev_preds_e5_ws={}
    # embedding=vectoriser.vectorise(['doc 1', 'doc 2', 'doc3'], batch_size=8)
    # FIX memory leak in vectoriser code
    # q_vect = vectoriser.encode_query('element 2')
    for claim in dev:
        print(claim['id'])
        citing_contexts = fcc.get_citing_text(claim['citekey'], chunksize=100)
        citing_vectors={}
        for e in citing_contexts:
            citing_vectors[e] = vectoriser.vectorise([citing_contexts[e]])[0]
        q_vect = vectoriser.encode_query(claim['claim'])[0]
        result = [(e, np.dot(citing_vectors[e], q_vect)) for e in citing_vectors]
        result.sort(key=lambda a: a[1], reverse=True)
        dev_preds_e5_ws[claim['id']] = [r[0] for r in result]


import sys
sys.path.append('../context24-main') 
import eval.task1_eval as ev

gold_labels = {x["id"]: x["findings"] for x in dev}
claim_citekeys = {x["id"]: x["citekey"] for x in dev}

for k in [-1500, -1000, -500]:
    dev_preds_e5_sem_chunk={}
    # embedding=vectoriser.vectorise(['doc 1', 'doc 2', 'doc3'], batch_size=8)
    # FIX memory leak in vectoriser code
    # q_vect = vectoriser.encode_query('element 2')
    for claim in dev:
        # print(claim['id'])
        citing_contexts = fcc.get_citing_text(claim['citekey'], chunksize=k)
        citing_vectors={}
        for e in citing_contexts:
            citing_vectors[e] = vectoriser.vectorise([citing_contexts[e]])[0]
        q_vect = vectoriser.encode_query(claim['claim'], task_instr='Given a claim, retrieve descriptions of figures and tables that support the claim')[0]
        result = [(e, np.dot(citing_vectors[e], q_vect)) for e in citing_vectors]
        result.sort(key=lambda a: a[1], reverse=True)
        dev_preds_e5_sem_chunk[claim['id']] = [r[0] for r in result]
    
    print(k)
    ev.run_eval(dev_preds_e5_sem_chunk, gold_labels, '../context24-main/figures-tables', claim_citekeys, debug=False)




if not run_new_only:
    print('Mistral e5:')
    ev.run_eval(dev_preds_e5, gold_labels, '../context24-main/figures-tables', claim_citekeys, debug=False)
    
    print('Mistral e5 web search:')
    ev.run_eval(dev_preds_e5_ws, gold_labels, '../context24-main/figures-tables', claim_citekeys, debug=False)