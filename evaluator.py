import csv
import nltk
import urllib2
import json
import math
import re

def get_vocabulary(lang, protocols, bound):
	voc = []

	with open('data/clean-labels-freeling-dedup.json') as data_file:    
		protocols_txt = json.load(data_file)

	# print len(protocols)
	for pr in protocols:
		p = protocols_txt[pr]
		if lang=='en':
			labels = p[0] 
		elif lang=='es':
			labels = p[1]

		for l in labels:
			words = [w[1] for w in l[3][0]]
			for w in words[:min(bound,len(words))]:
				if not w in voc:
					voc.append(w)
	return voc


def get_reference(lang, protocols, bound):

	reference = {}

	if lang == 'en':
		with open('data/trans_es_en_comp.json', 'r') as data:
			ox_dict = json.load(data)
		voc1 = get_vocabulary('en', protocols, bound)
		voc2 = get_vocabulary('es', protocols, bound)

	elif lang == 'es':
		with open('data/trans_en_es_comp.json', 'r') as data:
			ox_dict = json.load(data)
		voc1 = get_vocabulary('es', protocols, bound)
		voc2 = get_vocabulary('en', protocols, bound)


	for v in voc2:
		if v in ox_dict.keys():
			for w in [ww for ww in ox_dict[v] if ww in voc1]:
				if not v in reference.keys():
					reference[v] = []
				reference[v].append(w)


	return reference


def evaluate(alignment, lang, protocols, bound):
	reference = get_reference(lang, protocols, bound)
	good = {}
	bad = {}

	for v in reference.keys():
		good[v] = []
		bad[v] = []

		if v in alignment.keys():
			for w in alignment[v]:
				if w in reference[v]:
					good[v].append(w)
				else:
					bad[v].append(w)

	salg = sum([len(alignment[v]) for v in alignment.keys()])
	sref = sum([len(reference[v]) for v in reference.keys()])
	sgood = sum([len(good[v]) for v in good.keys()])
	sbad = sum([len(bad[v]) for v in bad.keys()])

	return float(sgood)/(float(sgood)+float(sbad))
