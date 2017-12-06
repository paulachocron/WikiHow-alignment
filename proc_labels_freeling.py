# -*- coding: utf-8 -*-
import re
import unicodedata
from io import open
from subprocess import call
import json
import freeling

def get_labels():
    protocols_en = []
    protocols_es = []

    with open('data/all_labels_de_duplicated.txt', encoding='utf-8') as f:
    # with open('all_labels.txt's) as f:
        content = f.readlines()
        f.close()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    labels_store = []
    iprot = -1
    jpair = 1
    labels = []
    
    # for l in content:
    for l in content:

        if not l: 
            continue
        elif l.startswith("#PROTOCOLPAIR"):
            print iprot        
            iprot += 1

            protocols_en.append([])
            protocols_es.append([])

        if l.startswith('#PAIR'):
            jpair = 1 - jpair
            # labels_store[iprot] = []
        elif not l.startswith("#"):

            words = l.split()
            # print l
            language_tag = words[0]
            type_tag = words[1]
            label_id = words[2]
            label = ' '.join(words[3:])

            ### this gets only the first sentence
            label = clean_label(label, language_tag)
            sentences = label.split('.')
            sentence = sentences[0]

            if language_tag == 'es':
                protocols_es[iprot].append((label_id, type_tag, sentence))
            elif language_tag == 'en':
                protocols_en[iprot].append((label_id, type_tag, sentence))
                
    return protocols_es, protocols_en


def build_freeling(lang):
    ##### Build resources
    FREELINGDIR = "/usr/local"

    DATA = FREELINGDIR+"/share/freeling/"
    LANG_ES="es"
    LANG_EN="en"

    freeling.util_init_locale("default")

    if lang == 'es':
        ##### Build Spanish analyzers
        op=freeling.maco_options("es")
        op.set_data_files( "", 
                           DATA + "common/punct.dat",
                           DATA + "es" + "/dicc.src",
                           DATA + "es" + "/afixos.dat",
                           "",
                           # "data/locutions_es_processed.dat", 
                           "", 
                           DATA + "es" + "/np.dat",
                           DATA + "es" + "/quantities.dat",
                           DATA + "es" + "/probabilitats.dat")
        # create analyzers
        op.MultiwordsDetection = True
        tk=freeling.tokenizer(DATA+"es"+"/tokenizer.dat")
        sp=freeling.splitter(DATA+"es"+"/splitter.dat")
        sid=sp.open_session()
        mf=freeling.maco(op)
        # activate mmorpho odules to be used in next call
        mf.set_active_options(False, True, True, True,  # select which among created 
                                True, True, False, True,  # submodules are to be used. 
                                True, True, True, True ) # default: all created submodules are used
        # create tagger
        tg=freeling.hmm_tagger(DATA+"es"+"/tagger.dat",True,2)
        
    elif lang == 'en':
    ##### Build English analyzers
        op=freeling.maco_options("en")
        op.set_data_files( "", 
                           DATA + "common/punct.dat",
                           DATA + "en" + "/dicc.src",
                           DATA + "en" + "/afixos.dat",
                           "",
                           # "data/locutions_en_processed.dat",
                           "",
                           DATA + "en" + "/np.dat",
                           DATA + "en" + "/quantities.dat",
                           DATA + "en" + "/probabilitats.dat")
        # create analyzers
        tk=freeling.tokenizer(DATA+"en"+"/tokenizer.dat")
        sp=freeling.splitter(DATA+"en"+"/splitter.dat")
        sid=sp.open_session()
        mf=freeling.maco(op)
        # activate mmorpho odules to be used in next call
        mf.set_active_options(False, True, True, True,  # select which among created 
                                True, True, False, True,  # submodules are to be used. 
                                True, True, True, True ) # default: all created submodules are used
        # create tagger
        tg=freeling.hmm_tagger(DATA+"en"+"/tagger.dat",True,2)
 
    return tk, sp, sid, mf, tg


def process_labels():

    tk_es, sp_es, sid_es, mf_es, tg_es = build_freeling('es')
    tk_en, sp_en, sid_en, mf_en, tg_en = build_freeling('en')

    protocols_es, protocols_en = get_labels()

    labels_store = []

    for iprot in range(len(protocols_en)):
        labels_store.append([[],[]])

        for language_tag in ['en', 'es']:
            if language_tag == 'en':
                prot = protocols_en[iprot]
                jpair = 0
            else:
                prot = protocols_es[iprot]
                jpair = 1    
            
            for l in prot:
                label = l[2]
                type_tag = l[1]
                label_id = l[0]
                label_pr = None

                # print label
            

                if language_tag == 'es':
                    label_pr = parse_label(language_tag,type_tag,label, tk_es, sp_es, sid_es, mf_es, tg_es)
                if language_tag == 'en':
                    label_pr = parse_label(language_tag,type_tag,label, tk_en, sp_en, sid_en, mf_en, tg_en)
                # print "label before processing {}".format(label_pr)            
                labels_store[iprot][jpair].append((language_tag, type_tag, label_id, label_pr))

    return labels_store

def process_labels_testing():

    tk_es, sp_es, sid_es, mf_es, tg_es = build_freeling('es')
    tk_en, sp_en, sid_en, mf_en, tg_en = build_freeling('en')

            
    label = "Mantequilla"
    language_tag = 'es'
    type_tag = 'up'
    
    label = clean_label(label, language_tag)
    if language_tag == 'es':
        label_pr = parse_label(language_tag,type_tag,label, tk_es, sp_es, sid_es, mf_es, tg_es)
    if language_tag == 'en':
        label_pr = parse_label(language_tag,type_tag,label, tk_en, sp_en, sid_en, mf_en, tg_en)
    print label
    print label_pr
    return label_pr


def lemmatizer():

    tk_es, sp_es, sid_es, mf_es, tg_es = build_freeling('es')
    tk_en, sp_en, sid_en, mf_en, tg_en = build_freeling('en')

    protocols_es, protocols_en = get_labels()

    labels_store = {}


    for iprot in range(len(protocols_en)):
        labels_store[iprot] = [[],[]]

        for language_tag in ['en', 'es']:
            if language_tag == 'en':
                tk, sp, sid, mf = (tk_en, sp_en, sid_en, mf_en)
                prot = protocols_en[iprot]
                jpair = 0
            else:
                tk, sp, sid, mf = (tk_es, sp_es, sid_es, mf_es)
                prot = protocols_es[iprot]
                jpair = 1    
            
            for l in prot:
                label = l[2]
                type_tag = l[1]
                label_id = l[0]
                # print label
                label = clean_label(label)
                label_pr = None
     
                tkd = tk.tokenize(label)
                spd = sp.split(sid,tkd, True)
                lf = mf.analyze(spd)
                lemmatized = []
                for ws in lf:
                    for w in ws.get_words():
                        lemmatized.append(w.get_lemma())
                lemlab = " ".join(lemmatized)
                            
                labels_store[iprot][jpair].append((language_tag, label_id, lemlab))

    return labels_store


def clean_label(label, language_tag='en'):
    #remove parenthesis
    label = label.lower()
    label = re.sub(r'[0-9]([.,][0-9])?', ' ', label)
    label = re.sub(r'[.,][0-9]', ' ', label)
    label = re.sub(r'[0-9]([x][0-9])?', ' ', label)

    # label = re.sub(r'\([^)]*\)', '', label)
    label = re.sub(r'\)', ' ', label)
    label = re.sub(r'\(', ' ', label)
    # remove numbers
    # label = re.sub(r'([#199;&#209;])+', '', label)
    # remove most common measures
    if language_tag=='en':
        centim = ' centimeters '
        milil = ' mililiters '
        lib = ' pounds '
        grm = ' grams '
        ounc = ' ounces '

    else:
        centim = ' centimetros '
        milil = ' mililitros '
        lib = ' libras '
        grm = ' gramos '
        ounc = ' onzas '

    label = re.sub(r'tbsp(\.)?|tbs(\.)?', ' tablespoon ', label)
    label = re.sub(r'tsp(\.)?', ' teaspoon ', label)
    label = re.sub(r' oz(\.)?', ounc, label)

    label = re.sub(r' cm(\.)?|cm\.', centim, label)
    label = re.sub(r' approx(\.)?', ' approximately ', label)
    label = re.sub(r' g(\.)? | gr(\.)? | grs(\.)? ', grm, label)
    label = re.sub(r' ml(\.)?', milil, label)
    label = re.sub(r' lb(\.)?|lb\.', lib, label)
    label = re.sub(r' qt(\.)?|qt\.', ' quarter ', label)
    label = re.sub(r'e\.g\.|ej\.|/', '', label)
    label = re.sub(r'ºC|ºc|ºF|ºf', '', label)
    label = re.sub(r'\xba,C|\xba,c|\xba,F|\xba,f', '', label)
    label = re.sub(r'cdts(\.)?|cdtas(\.)?|cdta(\.)?|cdt(\.)?', ' cucharaditas ', label)
    label = re.sub(r'cdas(\.)?|cda(\.)?', ' cucharadas ', label)
    label = re.sub(r'\'re', ' are', label)
    label = re.sub(r'º', '', label)
    label = re.sub(r'-', ' ', label)


    # label.replace('\xba,', '')
    # label = re.sub(r' g ', 'grams', label)
    label = ''.join([i for i in label if not i.isdigit()])
    #remove accents 
    # label = ''.join(c for c in unicodedata.normalize('NFD', label) if unicodedata.category(c) != 'Mn')
    # unicodedata.normalize('NFKD', label).encode('ascii','ignore')
    label = label.lower()
    # print "clean label {}".format(label)
    # label2 = label.encode('ascii','ignore')
    return label


def clean_label_dots(label):
    #remove parenthesis
    label = label.lower()

    label = re.sub(r'[.,][0-9]', ',5', label)
    # remove most common measures
    # label = re.sub(r' (oz(\.)?|cm(\.)?| fl(\.)?|g\.|ml(\.)?)', '', label)
    label = re.sub(r'tbsp(\.)?|tbs(\.)?|cdas(\.)', ' tablespoon ', label)
    label = re.sub(r'tsp(\.)?', ' teaspoon ', label)
    label = re.sub(r' oz(\.)?', ' ounces ', label)
    label = re.sub(r' cm(\.)?|cm\.', ' centimeters ', label)
    label = re.sub(r' g\.| gr\.| grs.| g ', ' grams ', label)
    label = re.sub(r' ml(\.)?', ' mililiters ', label)
    label = re.sub(r' lb(\.)?|lb\.', ' pounds ', label)
    label = re.sub(r' qt(\.)?|qt\.', ' quarter ', label)
    label = re.sub(r'e\.g\.|ej\.|/', '', label)
    label = re.sub(r' approx(\.)?', ' approximately ', label)
    # label = re.sub(r'ºC|ºc|ºF|ºf', '', label)
    label = re.sub(r'\xba,C|\xba,c|\xba,F|\xba,f', '', label)
    label = re.sub(r'cdts(\.)?|cdtas(\.)?|cdta(\.)?|cdt(\.)?', 'cucharadita', label)
    label = re.sub(r'cdts(\.)?|cdt(\.)?|cdtas(\.)?|cdta(\.)?', 'cucharadita', label)
    label = re.sub(r'cdas(\.)?|cda(\.)?', 'cucharada', label)
    label = re.sub(r'\'re', ' are', label)
    label = re.sub(r'º', '', label)
    label = re.sub(r'-', ' ', label)


    # label.replace('\xba,', '')
    # label = re.sub(r' g ', 'grams', label)
    
    return label


def parse_label(language_tag,type_tag,label, tk, sp, sid, mf, tg):

    # tag each sentence
    # Some cleaning
    sentences = label.split('.')
    sentences_res = []
    names = []

    for s in sentences:
        # print s
        tkd = tk.tokenize(s)
        # if s:
        #     names.extend(find_noun_phrases(s))
        spd = sp.split(sid,tkd, True)
        lf = mf.analyze(spd)
        lf = tg.analyze(lf)
       
        words = []
        
        for w in lf:
            for ws in w.get_words():

                if len(ws.get_form())>2 and not u'º' in ws.get_form():

                    if language_tag=='es':
                        if ws.get_tag()[0] =='V' or ws.get_tag()[0] == 'A' or ws.get_tag()[0]=='N' or ws.get_tag()[0]=='R':
                            words.append([ws.get_form(),ws.get_lemma().replace('_',' '), ws.get_tag()])
                        # elif ws.get_tag()=='Zp':
                        #      words.append([ws.get_form().replace('_',' '),ws.get_form().replace('_',' '), ws.get_tag()])
                    elif language_tag=='en':
                        if ws.get_tag()[0]=='V' or ws.get_tag()[0]=='J' or ws.get_tag()[0]=='N' or ws.get_tag()[0]=='R':
                            words.append([ws.get_form(),ws.get_lemma().replace('_',' '), ws.get_tag()])
                        # elif ws.get_tag()=='Zp':
                        #      words.append([ws.get_form().replace('_',' '),ws.get_form().split('_')[-1], ws.get_tag()])

        for w in words:
            if w[0] == 'cupcakes':
                w[1] = 'cupcake' 
            if w[0] == 'pancita':
                w[1] = 'panza' 
            if w[0] == 'milimeters':
                w[1] = 'milimeter'            
            if w[0] == 'muffins':
                w[1] = 'muffin'
            if w[0] == 'doritos':
                w[1] = 'dorito'
            if w[0] == 'brownies':
                w[1] = 'brownie'
            if w[1] == 'bayo':
                w[1] = 'baya'
            if w[0] == 'vegana' or w[0] == 'veganos' or w[0] == 'veganas':
                w[1] = 'vegano'
            if w[0] == 'nata' or w[0] == 'burrito' or w[0] == 'easter' or w[0] == 'boba' or w[0]=='ramito' or w[0]=='helado' or w[0]=='granada' or w[0]=='chalota':
                w[1] = w[0]
            if w[0][:7] == 'apropiad':
                w[1] = 'apropiado'
            if w[1] == 'vas':
                w[1] = 'vaso'
            if w[1] == 'morar':
                w[1] = 'morado'
            if w[0][:3] == 'mix':
                w[1] = 'mix'
            if any(d.isdigit() for d in w[1]):
                w[1] = w[0]

        sentences_res.append(words)

    return sentences_res


