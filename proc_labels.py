# -*- coding: utf-8 -*-
import re
import nltk
import unicodedata
from io import open
# from nltk.internals import find_jars_within_path
from nltk.tag.stanford import POSTagger
from subprocess import call
import json
import os
from pycorenlp import StanfordCoreNLP


def test():
    nlp = StanfordCoreNLP('http://localhost:9000')
    # text = ('Juan y Maria fueron al mar')
    text = ('John and Mary went for a walk')
    # If the server is started without any language properties, but the properties are loaded,
    # they can be specified directly in the properties.
    output = nlp.annotate(text, properties={
  'annotators': 'tokenize,pos',
  'outputFormat': 'json'})
  # "tokenize.language":"es",
  # "pos.model":"edu/stanford/nlp/models/pos-tagger/spanish/spanish-distsim.tagger"})
    print(output)

def get_labels():
    protocols = []
    with open('all_labels.txt', encoding='utf-8') as f:
    # with open('all_labels.txt's) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    labels_store = []
    iprot = -1
    jpair = 1
    labels = []
    nlp = StanfordCoreNLP('http://localhost:9000')
    for l in content:
        # if iprot>=1:
        #     break
        if not l: 
            continue
        elif l.startswith("#PROTOCOLPAIR"):
            print iprot        
            iprot += 1
            labels_store.append([0,1])
        elif l.startswith('#PAIR'):
            jpair = 1 - jpair
            labels_store[iprot][jpair] = []
        elif not l.startswith("#"):
            words = l.split()
            language_tag = words[0]
            type_tag = words[1]
            label_id = words[2]
            label = ' '.join(words[3:])

            label_pr = label
            # if language_tag == 'es':
            if language_tag == 'es' or language_tag == 'en' :
                # labels.append(label)
                label_pr = parse_label(language_tag,type_tag,label,nlp)

                labels_store[iprot][jpair].append((language_tag, type_tag, label_id, label_pr))

    return labels_store

def parse_label(language_tag,type_tag,label, nlp):
    # print unicode(label)
    #remove parenthesis
    label = re.sub(r'\([^)]*\)', '', label)
    # remove numbers
    label = re.sub(r'([&#199;&#209;])+', '', label)
    # remove most common measures
    label = re.sub(r'oz(\.)?|cm(\.)?|onzas|fl(\.)?|tsp(\.)?|tbsp(\.)?|g\.|ml(\.)?', '', label)
    label = re.sub(r'e\.g\.|ej\.|/', '', label)
    label = ''.join([i for i in label if not i.isdigit()])
    #remove accents 
    # label = ''.join(c for c in unicodedata.normalize('NFD', label) if unicodedata.category(c) != 'Mn')
    unicodedata.normalize('NFKD', label).encode('ascii','ignore')
    label = label.lower()
    label2 = label.encode('ascii','ignore')


    st = None
    
    sentences = []
    if language_tag == 'en':
        output = nlp.annotate(label2, properties={
      'annotators': 'tokenize,pos',
      'outputFormat': 'json'})

        sentences = []
        # for each sentence:
        for sentence in output['sentences']:
            verbs = []
            nouns = []
            # for each tokenized word:
            for pw in sentence['tokens']:
                if pw['pos'] == 'VBP' or pw['pos'] == 'VB':
                    verbs.append(pw['word'])
                elif pw['pos'][0] == 'N':
                    nouns.append(pw['word'])

            sentences.append((verbs,nouns))
            # if we want sentences as a dictionary
            # sentences[sentence['index']] = (verbs,nouns)

    if language_tag == 'es':
        # clean the output file
        with open('spanish-output.txt', 'w+') as outputf:

        # If the server is started without any language properties, but the properties are loaded,
        # they can be specified directly in the properties.
            output = nlp.annotate(label2, properties={
          'annotators': 'tokenize,pos',
          'outputFormat': 'json',
         "tokenize.language":"es",
        "pos.model":"edu/stanford/nlp/models/pos-tagger/spanish/spanish-distsim.tagger"})

        sentences = []
        # for each sentence:
        for sentence in output['sentences']:
            verbs = []
            nouns = []
            # for each tokenized word:
            for pw in sentence['tokens']:
                if pw['pos'][:3] == 'vmm' or pw['pos'][:3] == 'vmi':
                    verbs.append(pw['word'])
                elif pw['pos'][0] == 'n':
                    nouns.append(pw['word'])

            sentences.append((verbs,nouns))
    return sentences


store = get_labels()
# print store
with open('clean-labels.txt', 'w+') as outfile:
    outfile.write(unicode(json.dumps(store, ensure_ascii=False)))


# ok, maybe it will be better to do all this in the end.