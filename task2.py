import re
import os
import torch
import pandas as pd
import json
import argparse
import nltk
import requests
from rank_bm25 import BM25Okapi
import openai
import spacy
from openai import OpenAI
from openai import AzureOpenAI
from transformers import set_seed
from eval import task2_eval as e
api_endpoint = 'API_KEY'
function_name = 'call_claude_service'
function_url = f'{api_endpoint}/{function_name}'

spacy_model = 'en_core_web_md'
from nltk.tokenize import word_tokenize
max_tokens = 3000

nlp = spacy.load(spacy_model)

def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def extract_clauses(sentence):
    doc = nlp(sentence)
    clauses = []
    for sent in doc.sents:
        root = sent.root
        clause = [token.text for token in root.subtree]
        clauses.append(" ".join(clause))
    return clauses


def count_tokens(prompt):
    # Use word_tokenize from nltk to count tokens
    tokens = word_tokenize(prompt)
    return len(tokens)


def find_longest_common_substring(s1, s2):
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest]


def find_longest_common_phrase(sentence1, sentence2):
    # Split sentences into phrases
    phrases_sentence1 = re.findall(r'\b[\w\s]+\b', sentence1)
    phrases_sentence2 = re.findall(r'\b[\w\s]+\b', sentence2)

    # Initialize longest common phrase
    longest_common_phrase = ""

    # Iterate over phrases in first sentence
    for phrase1 in phrases_sentence1:
        # Iterate over phrases in second sentence
        for phrase2 in phrases_sentence2:
            # Find the longest common substring between phrases
            common_substring = find_longest_common_substring(phrase1, phrase2)
            # If the common substring is longer than the current longest common phrase, update it
            if len(common_substring) > len(longest_common_phrase):
                longest_common_phrase = common_substring

    return longest_common_phrase


def find_longest_common_phrase_in_sentence_list(sentence_list):
    longest_common_phrases = []
    for i in range(len(sentence_list)):
        for j in range(i + 1, len(sentence_list)):
            longest_common_phrase = find_longest_common_phrase(sentence_list[i], sentence_list[j])
            longest_common_phrases.append((sentence_list[i], sentence_list[j], longest_common_phrase))
    return longest_common_phrases


def upload_fine_tuning_data(client, data_path):
  uploaded_file = client.files.create(
    file=open(data_path, "rb"),
    purpose='fine-tune'
  )

  return uploaded_file


def prepare_train_data_gpt(dataset, args):
    train_data = []
    data = pd.read_csv('task2_files/context24.csv')

    with open('task2_files/full_texts.json', "r") as file:
        data_not_clean = json.load(file)
    with open("task2_files/train_data.jsonl", "w", encoding="utf8") as file:
        for sample in dataset['train']:
            cite_key, claim, evidence = sample['citekey'], sample['claim'], sample['context']
            if cite_key + '.pdf' in set(data['doc_path']):
                full_text_ = ' '.join(data[data['doc_path'] == sample['citekey'] + '.pdf']['paragraph'])
            else:
                full_text_ = data_not_clean[cite_key]
            message = {"messages": [{"role": "system", "content": f"[Given Article Start]\n{full_text_}\n[Given Article End]"},
                                         {"role": "user", "content": f"What are evidence phrases for given claim {claim}"},
                                          {"role": "assistant", "content": " ".join(evidence)}
                           ]}
            json.dump(message, file)
            file.write('\n')


def openai_get_prompt(article, claim):
    instruction = f"Please extract evidence clauses from the given article for given claim {claim}" + \
                  f"[Given Article Start]\n{article}\n[Given Article End]\n\n"

    return instruction


def open_ai_generate_response(prompt, client, deployment_name):
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)

    response_text = response.choices[0].message.content

    return response_text


def claude_get_prompt(article, claim):
    instruction = f"Please extract evidence sentences from the given article for given claim {claim}" + \
                f"[Given Article Start]\n{article}\n[Given Article End]\n\n"


    basic_structure = f"Human: {instruction}" + f"\n\n\n\n\nAssistant:"

    return basic_structure


def claude_get_response(prompt, version='v1'):
    modelID = 'anthropic.claude-v2'
    if version == 'v1':
        modelID = 'anthropic.claude-instant-v1'

    payload = {
        'modelID': modelID,
        'full_prompt': prompt,
        'temperature': 0.1,  # Example value, replace with your actual value
        'top_p': 0.95  # Example value, replace with your actual value
    }

    # Make a GET request to the function URL
    response = requests.post(function_url, data=payload)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        return data
    else:
        # Print an error message if the request was unsuccessful
        print(f"Error: {response.status_code} - {response.text}")


def llm_prediction(dataset, client, args):

    for sample in dataset['test']:
        cite_key, claim = sample['citekey'], sample['claim']
        if cite_key + '.pdf' in set(dataset['clean_full_text']['doc_path']):
            full_text_ = dataset['clean_full_text'][dataset['clean_full_text']['doc_path'] == cite_key + '.pdf'][
                'paragraph']
            full_text_ = ' '.join(full_text_)
        else:
            print(cite_key)
            full_text_ = dataset['full_text'][cite_key]
        if args.model == 'claude':
            'try'
            prompt = claude_get_prompt(full_text_, claim)
            respond = claude_get_response(prompt)['claude_response']
        elif args.model =='gpt':

            if False: #not finetuned
                client = AzureOpenAI(
                    api_key="API_KEY",
                    api_version="2024-02-01",
                    azure_endpoint=".................."
                )
                deployment_name = 'firstcontact-gpt4-turbo'
                prompt = openai_get_prompt(full_text_, claim)
                respond = open_ai_generate_response(prompt, client, deployment_name)
            else:
                prompt_tokens = count_tokens(full_text_)
                if prompt_tokens > 8000:
                    # Split the prompt into tokens
                    prompt_list = word_tokenize(full_text_)

                    # Truncate the prompt to the specified max_tokens
                    full_text_ = ' '.join(prompt_list[:8000])

                t = {"messages": [
                    {"role": "system", "content": f"[Given Article Start]\n{full_text_}\n[Given Article End]"},
                    {"role": "user",
                     "content": "What are evidence clauses from the article for given claim " + claim}
                ]
                }
                completion = client.chat.completions.create(
                    model="ft:gpt-3.5-turbo-0125:csiro::9UuuCLLd",
                    messages=t['messages']
                )
                respond = completion.choices[0].message.content
        sample['context'] = [respond]

    with open(args.out_file, "w", encoding="utf8") as file:
        json.dump(dataset['test'], file)


def bm25_prediction(dataset, args):
    if args.clean:
        for sample in dataset['test']:
            cite_key, claim = sample['citekey'], sample['claim']
            if args.page:
                if args.neighbor:
                    if cite_key + '.pdf' in set(dataset['clean_full_text']['doc_path']):
                        full_text_ = dataset['clean_full_text'][dataset['clean_full_text']['doc_path'] == sample['citekey'] + '.pdf']
                        pages, all_sentences = [], []
                        for page, paragraph in zip(full_text_['page_number'], full_text_['paragraph']):
                            sent_text = nltk.sent_tokenize(paragraph)
                            all_sentences.extend(sent_text)
                            pages.extend([page] * len(sent_text))
                        tokenized_corpus = [doc.split() for doc in all_sentences]
                        bm25 = BM25Okapi(tokenized_corpus)
                        sentences = bm25.get_top_n(
                            claim.split(), all_sentences, n=args.num_sentences
                        )
                        conc, left, right = [], [], []
                        for item in sentences:
                            index = all_sentences.index(item)
                            if index != 0:
                                left_index = index - 1
                                left.append(all_sentences[left_index] + ' (p. ' + str(pages[left_index]) + ') ')
                            if index != len(all_sentences) - 1:
                                right_index = index + 1
                                right.append(all_sentences[right_index] + ' (p. ' + str(pages[right_index]) + ') ')
                            item += ' (p. ' + str(pages[index]) + ') '
                            conc.append(item)
                            if args.neighbor == 'left' and len(left) > 0:
                                conc.append(left[-1])
                            elif args.neighbor == 'right' and len(left) > 0:
                                conc.append(right[-1])

                        sample['context'] = [' '.join(conc)]
                    else:
                        full_text_ = dataset['full_text'][cite_key]
                        all_sentences = nltk.sent_tokenize(full_text_)

                        tokenized_corpus = [doc.split() for doc in all_sentences]
                        bm25 = BM25Okapi(tokenized_corpus)
                        sentences = bm25.get_top_n(
                            claim.split(), all_sentences, n=args.num_sentences
                        )
                        conc, left, right = [], [], []
                        for item in sentences:
                            index = all_sentences.index(item)
                            if index != 0:
                                left_index = index - 1
                                left.append(all_sentences[left_index] )
                            if index != len(all_sentences) - 1:
                                right_index = index + 1
                                right.append(all_sentences[right_index])

                            conc.append(item)
                            if args.neighbor == 'left' and len(left) > 0:
                                conc.append(left[-1])
                            elif args.neighbor == 'right' and len(left) > 0:
                                conc.append(right[-1])

                        sample['context'] = [' '.join(conc)]
                else:
                    if cite_key + '.pdf' in set(dataset['clean_full_text']['doc_path']):
                        full_text_ = dataset['clean_full_text'][dataset['clean_full_text']['doc_path'] == sample['citekey'] + '.pdf']
                        pages, all_sentences = [], []
                        for page, paragraph in zip(full_text_['page_number'], full_text_['paragraph']):
                            sent_text = nltk.sent_tokenize(paragraph)
                            all_sentences.extend(sent_text)
                            pages.extend([page] * len(sent_text))
                        tokenized_corpus = [doc.split() for doc in all_sentences]
                        bm25 = BM25Okapi(tokenized_corpus)
                        sentences = bm25.get_top_n(
                            claim.split(), all_sentences, n=args.num_sentences
                        )
                        sentences_page = []
                        for item in sentences:
                            index = all_sentences.index(item)
                            #item += ' (p. '+str(pages[index]) + ') '
                            sentences_page.append(item)
                        #sample['context'] = [' '.join(sentences_page)]
                        sample['context'] = sentences_page
                    else:
                        full_text_ = dataset['full_text'][cite_key]
                        all_sentences = nltk.sent_tokenize(full_text_)

                        tokenized_corpus = [doc.split() for doc in all_sentences]
                        bm25 = BM25Okapi(tokenized_corpus)
                        sentences = bm25.get_top_n(
                            claim.split(), all_sentences, n=args.num_sentences
                        )
                        #sample['context'] = [' '.join(sentences)]
                        sample['context'] = sentences

            else:
                if cite_key+'.pdf' in set(dataset['clean_full_text']['doc_path']):
                    full_text_ = dataset['clean_full_text'][dataset['clean_full_text']['doc_path'] == cite_key + '.pdf']['paragraph']
                    full_text_ = ' '.join(full_text_)
                else:
                    full_text_ = dataset['full_text'][cite_key]
                sentences = nltk.sent_tokenize(full_text_)

                tokenized_corpus = [doc.split() for doc in sentences]
                bm25 = BM25Okapi(tokenized_corpus)
                sentences = bm25.get_top_n(
                    claim.split(), sentences, n=args.num_sentences
                )
                sample['context'] = [' '.join(sentences)]
        with open(args.out_file, "w", encoding="utf8") as file:
            json.dump(dataset['test'], file)


def bm25_claude_prediction(dataset, args):
    for sample in dataset['test']:
        cite_key, claim = sample['citekey'], sample['claim']

        if cite_key + '.pdf' in set(dataset['clean_full_text']['doc_path']):
            full_text_ = dataset['clean_full_text'][dataset['clean_full_text']['doc_path'] == sample['citekey'] + '.pdf']
            pages, all_sentences = [], []
            for page, paragraph in zip(full_text_['page_number'], full_text_['paragraph']):
                sent_text = nltk.sent_tokenize(paragraph)
                all_sentences.extend(sent_text)
                pages.extend([page] * len(sent_text))
            tokenized_corpus = [doc.split() for doc in all_sentences]
            bm25 = BM25Okapi(tokenized_corpus)
            sentences = bm25.get_top_n(
                claim.split(), all_sentences, n=args.num_sentences
            )
            sentences_page = []
            for item in sentences:
                index = all_sentences.index(item)
                item += ' (p. ' + str(pages[index]) + ') '
                sentences_page.append(item)
            sample['context'] = [' '.join(sentences_page)]
        else:
            full_text_ = dataset['full_text'][cite_key]
            all_sentences = nltk.sent_tokenize(full_text_)

            tokenized_corpus = [doc.split() for doc in all_sentences]
            bm25 = BM25Okapi(tokenized_corpus)
            sentences = bm25.get_top_n(
                claim.split(), all_sentences, n=args.num_sentences
            )
        prompt = claude_get_prompt(' '.join(sentences), claim)
        respond = claude_get_response(prompt)['claude_response']
        sample['context'] = [respond]
    with open(args.out_file, "w", encoding="utf8") as file:
        json.dump(dataset['test'], file)


def rule_based(dataset, args):
    evidences = []
    nlp = spacy.load(spacy_model)
    for sample in dataset['train']:
        cite_key, claim, evidence = sample['citekey'], sample['claim'], sample['context']
        evidences.extend(evidence)

    patterns = find_longest_common_phrase_in_sentence_list(evidences)
    commons = []
    for pattern in patterns:
        commons.append(pattern[2])
    pos_gold_list = ['NNS', 'NN', 'ADJ', 'VB']
    for sample in dataset['test']:
        cite_key, claim = sample['citekey'], sample['claim']
        if cite_key + '.pdf' in set(dataset['clean_full_text']['doc_path']):
            full_text_ = dataset['clean_full_text'][dataset['clean_full_text']['doc_path'] == sample['citekey'] + '.pdf']
            pages, all_sentences = [], []
            for page, paragraph in zip(full_text_['page_number'], full_text_['paragraph']):
                if False:
                    sent_text = nltk.sent_tokenize(paragraph)
                    all_sentences.extend(sent_text)
                    pages.extend([page] * len(sent_text))
                else:
                    doc = nlp(paragraph)
                    clauses = []
                    for sent in doc.sents:
                        root = sent.root
                        clause = [token.text for token in root.subtree]
                        clauses.extend(clause)
                    all_sentences.extend(clauses)
                    pages.extend([page] * len(clauses))
            predic = []

            for sent in all_sentences:
                for item in set(commons):
                    if item in sent:
                        pos_list = [pos for word, pos in nltk.pos_tag(item.split())]
                        if set(pos_gold_list).intersection(pos_list):
                            predic.append(sent)

            tokenized_corpus = [sent.split() for sent in predic]
            bm25 = BM25Okapi(tokenized_corpus)
            pred = bm25.get_top_n(
                claim.split(), predic, n=5
            )
            final = []
            for item in pred:
                index = all_sentences.index(item)
                item += ' (p. ' + str(pages[index]) + ') '
                final.append(item)

            sample['context'] = [' '.join(list(set(final)))]

        else:
            full_text_ = dataset['full_text'][cite_key]
            all_sentences = nltk.sent_tokenize(full_text_)

            for sent in all_sentences:
                for item in set(commons):
                    if item in sent:
                        pos_list = [pos for word, pos in nltk.pos_tag(item.split())]
                        if set(pos_gold_list).intersection(pos_list):
                            predic.append(sent)

            tokenized_corpus = [sent.split() for sent in predic]
            bm25 = BM25Okapi(tokenized_corpus)
            pred = bm25.get_top_n(
                claim.split(), predic, n=10
            )
            final = []
            for item in pred:
                index = all_sentences.index(item)
                item += ' (p. ' + str(pages[index]) + ') '
                final.append(item)

            sample['context'] = [' '.join(list(set(final)))]


    with open(args.out_file, "w") as file:
        json.dump(dataset['test'], file)

def main(args):
    client = OpenAI(api_key="API_KEY")
    set_seed(args.seed)
    train_file = os.path.join(args.data_dir, "task2_train.json")
    #dev_file = os.path.join(args.data_dir, "dev.txt")
    test_file = os.path.join(args.data_dir, "task2-test_.json")
    clean_full_text_file = os.path.join(args.data_dir, "context24.csv")
    full_text_file = os.path.join(args.data_dir, "full_texts-test.json")
    # train dataset
    with open(train_file, "r", encoding="utf8") as file:
        train_samples = json.load(file)
    # test dataset
    with open(test_file, "r", encoding="utf8") as file:
        test_samples = json.load(file)
    #clean full text
    full_text_clean = pd.read_csv(clean_full_text_file)
    # full text
    with open(full_text_file, "r") as file:
        full_text = json.load(file)

    dataset = {'train': train_samples,
               'test': test_samples,
               'full_text': full_text,
               'clean_full_text': full_text_clean}

    if args.model == 'bm25':
        bm25_prediction(dataset, args)
    elif args.model in['gpt', 'claude']:
        llm_prediction(dataset, client, args)
    elif args.model == 'bm25+claude':
        bm25_claude_prediction(dataset, args)
    elif args.model == 'rule_based':
        rule_based(dataset, args)
    elif args.model == 'finetuning':
        prepare_train_data_gpt(dataset, args)

        uploaded_training_data = upload_fine_tuning_data(client, "task2_files/train_data.jsonl")

        client.fine_tuning.jobs.create(
            training_file=uploaded_training_data.id,
            model="gpt-3.5-turbo"
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_dir", default="task2_files/", type=str)
    parser.add_argument("--out_file", default="task2_files/bm25.json", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_sentences", default=5, type=int)
    parser.add_argument("--model", default="bm25", type=str) # claude, gpt, bm25
    parser.add_argument("--page", default=False, action="store_true")
    parser.add_argument("--clean", default=True, action="store_true")
    parser.add_argument("--neighbor", default=None, type=str)
    parsed_args = parser.parse_args()
    main(parsed_args)
