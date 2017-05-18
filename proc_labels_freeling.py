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
import freeling
# from pycorenlp import StanfordCoreNLP

def process_labels():

    ##### Build resources
    FREELINGDIR = "/usr/local"

    DATA = FREELINGDIR+"/share/freeling/"
    LANG_ES="es"
    LANG_EN="en"

    freeling.util_init_locale("default")

    ##### Build Spanish analyzers
    op_es=freeling.maco_options("es")
    op_es.set_data_files( "", 
                       DATA + "common/punct.dat",
                       DATA + "es" + "/dicc.src",
                       DATA + "es" + "/afixos.dat",
                       "",
                       DATA + "es" + "/locucions.dat", 
                       DATA + "es" + "/np.dat",
                       DATA + "es" + "/quantities.dat",
                       DATA + "es" + "/probabilitats.dat")
    # create analyzers
    tk_es=freeling.tokenizer(DATA+"es"+"/tokenizer.dat")
    sp_es=freeling.splitter(DATA+"es"+"/splitter.dat")
    sid_es=sp_es.open_session()
    mf_es=freeling.maco(op_es)
    # activate mmorpho odules to be used in next call
    mf_es.set_active_options(False, True, True, True,  # select which among created 
                            True, True, False, True,  # submodules are to be used. 
                            True, True, True, True ) # default: all created submodules are used
    # create tagger
    tg_es=freeling.hmm_tagger(DATA+"es"+"/tagger.dat",True,2)
    

    ##### Build English analyzers
    op_en=freeling.maco_options("en")
    op_en.set_data_files( "", 
                       DATA + "common/punct.dat",
                       DATA + "en" + "/dicc.src",
                       DATA + "en" + "/afixos.dat",
                       "",
                       DATA + "en" + "/locucions.dat", 
                       DATA + "en" + "/np.dat",
                       DATA + "en" + "/quantities.dat",
                       DATA + "en" + "/probabilitats.dat")
    # create analyzers
    tk_en=freeling.tokenizer(DATA+"en"+"/tokenizer.dat")
    sp_en=freeling.splitter(DATA+"en"+"/splitter.dat")
    sid_en=sp_en.open_session()
    mf_en=freeling.maco(op_en)
    # activate mmorpho odules to be used in next call
    mf_en.set_active_options(False, True, True, True,  # select which among created 
                            True, True, False, True,  # submodules are to be used. 
                            True, True, True, True ) # default: all created submodules are used
    # create tagger
    tg_en=freeling.hmm_tagger(DATA+"en"+"/tagger.dat",True,2)
 

    ####### Retrieve the labels
    protocols = []
    with open('data/all_labels.txt', encoding='utf-8') as f:
    # with open('all_labels.txt's) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    labels_store = []
    iprot = -1
    jpair = 1
    labels = []
    
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
            if language_tag == 'es':
                # labels.append(label)
                label_pr = parse_label(language_tag,type_tag,label, tk_es, sp_es, sid_es, mf_es, tg_es)
            if language_tag == 'en' :
                # labels.append(label)
                label_pr = parse_label(language_tag,type_tag,label, tk_en, sp_en, sid_en, mf_en, tg_en)

            labels_store[iprot][jpair].append((language_tag, type_tag, label_id, label_pr))

    return labels_store


def parse_label(language_tag,type_tag,label, tk, sp, sid, mf, tg):
    # Some cleaning
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
    # unicodedata.normalize('NFKD', label).encode('ascii','ignore')
    label = label.lower()
    label2 = label.encode('ascii','ignore')
    # label2 = label.encode('ascii','ignore')

    # Now tag each sentence
    sentences = label.split('.')
    sentences_res = []

    for s in sentences:
        # print s
        tkd = tk.tokenize(s)
        spd = sp.split(sid,tkd, True)
        lf = mf.analyze(spd)
        lf = tg.analyze(lf)
       
        words = []
        
        for w in lf:
            for ws in w.get_words():
                # print ws.get_form() + " " + ws.get_lemma() + " " + ws.get_tag()
                if language_tag=='es':
                    if ws.get_tag()[:4] =='VMIP'or ws.get_tag()[0]=='N':
                        words.append((ws.get_form(),ws.get_lemma(), ws.get_tag()))
                elif language_tag=='en':
                    if ws.get_tag()=='VB' or ws.get_tag()=='VBP' or ws.get_tag()[0]=='N':
                        words.append((ws.get_form(),ws.get_lemma(), ws.get_tag()))

        sentences_res.append(words)

    return sentences_res

store = process_labels()
# print store
with open('data/clean-labels-freeling.txt', 'w+') as outfile:
    outfile.write(unicode(json.dumps(store, ensure_ascii=False)))


# ok, maybe it will be better to do all this in the end.