#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:02:37 2024

@author: ryb003
"""

from rank_bm25 import BM25Okapi
import nltk
# nltk.download('punkt')
from nltk import word_tokenize, sent_tokenize

from semantic_text_splitter import TextSplitter

splitter = TextSplitter()

# Maximum number of characters in a chunk. Will fill up the
# chunk until it is somewhere in this range.
# chunks = splitter.chunks("your document text", chunk_capacity=(200,1000))

import os
import json
rootdir = '../context24-main/figures-tables'
docs2elements={}
for _, dirs, _ in os.walk(rootdir):
    for d in dirs:
        path = os.path.join(rootdir, d)
        for _, _, files in os.walk(path):
            docs2elements[d]=[f.replace(".png", "") for f in files if 'CAPTION' not in f]
import sys
this = sys.modules[__name__]
            
this.empty_counter=0
this.element_counter=0

with open('../context24-main/full_texts.json') as f:
    fulltexts = json.load(f)
    
with open('captions_update.json') as f:
    captions = json.load(f)
     
 
def create_index_of_captions_and_citing(chunksize=100):
    docXchunkIDs=[]
    corpus=[]
    for doc_id in fulltexts:
        citing_texts = get_citing_text(doc_id, chunksize=chunksize)
        texts=list(citing_texts.values())
        IDs=list(citing_texts.keys())
        docXchunkIDs.extend([doc_id+'##'+i for i in IDs])
        corpus.extend(texts)
    ids, captions,_ = create_index_of_captions()
    cap_dict={i:cap for i,cap in zip(ids, captions)}
    for i, fig_id in enumerate(docXchunkIDs):
        if fig_id in cap_dict:
            if corpus[i] != 'emmpptty':
                corpus[i] += ' ' + cap_dict[fig_id]
            else:
                corpus[i] = cap_dict[fig_id]
    for fig_id in cap_dict:
        if fig_id not in set(docXchunkIDs):
            docXchunkIDs.append(fig_id)
            corpus.append(cap_dict[fig_id])
            
    tokenised = [word_tokenize(t.lower()) for t in corpus]
    bm25 = BM25Okapi(tokenised)
    # tokenised_query = word_tokenize(claim['claim'].lower())
    return docXchunkIDs, corpus, bm25

def create_index_of_citing_chunks(chunksize=100):
    docXchunkIDs=[]
    corpus=[]
    for doc_id in fulltexts:
        citing_texts = get_citing_text(doc_id, chunksize=chunksize)
        texts=list(citing_texts.values())
        IDs=list(citing_texts.keys())
        docXchunkIDs.extend([doc_id+'##'+i for i in IDs])
        corpus.extend(texts)
    tokenised = [word_tokenize(t.lower()) for t in corpus]
    bm25 = BM25Okapi(tokenised)
    # tokenised_query = word_tokenize(claim['claim'].lower())
    return docXchunkIDs, corpus, bm25

def create_index_of_captions():
    docXchunkIDs=[]
    corpus=[]
    for caption_dict in captions:
        doc_id=caption_dict['doc_id']
        for fig_id in caption_dict['caption_content']:
            e = fig_id.strip()[:-1].replace('FIG', 'FIG ').replace('TAB', 'TAB ')
            text=caption_dict['caption_content'][fig_id]         
            docXchunkIDs.append(doc_id+'##'+e)
            corpus.append(text)
    tokenised = [word_tokenize(t.lower()) for t in corpus]
    bm25 = BM25Okapi(tokenised)
    # tokenised_query = word_tokenize(claim['claim'].lower())
    return docXchunkIDs, corpus, bm25

def oracle_ranker(ground_truth, doc_id, chunksize=100):
    citing_texts = get_citing_text(doc_id, chunksize=chunksize)
    result=[]
    for e in citing_texts:
        if len(citing_texts[e])>10 and e in ground_truth:
            result.append(e)
    return result

def create_index_of_all_chunks(chunksize=100):
    docXchunkIDs=[]
    corpus=[]
    for doc_id in fulltexts:
        doc=fulltexts[doc_id]
        if chunksize>0:
            words=doc.split()
            chunks = [' '.join(words[i:i+chunksize]) for i in range(0,len(words),chunksize)]
        elif chunksize==0:
            chunks = sent_tokenize(doc)
        else:
            maxchar=-1*chunksize
            chunks = splitter.chunks(doc, maxchar)
            
        IDs=list(range(len(chunks)))
        docXchunkIDs.extend([doc_id+'##'+str(i) for i in IDs])
        corpus.extend(chunks)
    tokenised = [word_tokenize(t.lower()) for t in corpus]
    bm25 = BM25Okapi(tokenised)
    return docXchunkIDs, corpus, bm25

def score_passages(q, doc_id, index_ids, index):
    # elements=[]
    bm25_ids=[]
    for i, ident in enumerate(index_ids):
        split=ident.split('##')
        if split[0] == doc_id:
            # elements.append(split[1])
            bm25_ids.append(i)
    scores=index.get_batch_scores(word_tokenize(q.lower()), bm25_ids)
    l=list(zip(bm25_ids, scores))
    l.sort(key=lambda a: a[1], reverse=True)
    return [e[0] for e in l]
    # return elements, scores

def score_elements(q, doc_id, index_ids, index):
    elements=[]
    bm25_ids=[]
    for i, ident in enumerate(index_ids):
        # print(ident)
        split=ident.split('##')
        if split[0] == doc_id:
            elements.append(split[1])
            bm25_ids.append(i)
    scores=index.get_batch_scores(word_tokenize(q.lower()), bm25_ids)
    l=list(zip(elements, scores))
    l.sort(key=lambda a: a[1], reverse=True)
    return [e[0] for e in l]
    # return elements, scores
 
def get_citing_text(doc_id, chunksize=100):
    doc=fulltexts[doc_id]
    if doc_id in docs2elements:
        elements=docs2elements[doc_id]
    else:
        return {}
    citing_texts={}
    
    words=doc.split()
    if chunksize>0:
        words=doc.split()
        chunks = [' '.join(words[i:i+chunksize]) for i in range(0,len(words),chunksize)]
    elif chunksize==0:
        chunks = sent_tokenize(doc)
    else:
        maxchar=-1*chunksize
        chunks = splitter.chunks(doc, maxchar)
    
    this.element_counter += len(elements)
    
    for e in elements:
        element = e.strip()+ ' '
        if 'FIG ' in element and 'FIGURE' in element:
            element=element.replace('FIGURE', '')
        aliases=[element]
        if 'FIG' in element:
            aliases.append(element.replace('FIG', 'fig.'))
            aliases.append(element.replace('FIG', 'fig .'))
            aliases.append(element.replace('FIG', 'figure'))
            aliases.append(element.replace('FIG', 'f.'))
            aliases.append(element.replace('FIG', 'f .'))
        if 'TAB' in element:
            aliases.append(element.replace('TAB', 'tab.'))
            aliases.append(element.replace('TAB', 'tab .'))
            aliases.append(element.replace('TAB', 'table'))
            aliases.append(element.replace('TAB', 't.'))
            aliases.append(element.replace('TAB', 't .'))
        
        citing_text=''
        for chunk in chunks:
            for alias in aliases:
                if alias.lower() in chunk.lower():
                    citing_text+= ' ' + chunk
                    break
        if len(citing_text)==0:
            citing_text='emmpptty'
            this.empty_counter+=1
            # print(e + ' not found in '+ doc_id)
            # print([a.lower() for a in aliases])
            # print()
        citing_texts[e]=citing_text
    return citing_texts

# goyalEffectsSensemakingTranslucence2016

# q= 'low prevalence of EBPR references in state statutes and regulations pertaining to behavioral health care; 20 states had a total of 33 mandates that referenced an EBPR'
# ids, docs, idx=create_index_of_citing_chunks()
# e=score_elements(q, 'lee2022references', ids, idx)
