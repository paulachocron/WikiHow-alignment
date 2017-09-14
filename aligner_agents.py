	#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, getopt
from multiprocessing import Process, Pipe, Queue
import random
from nltk import bigrams
from operator import itemgetter
import numpy as np
import itertools
import json
import time
from evaluator import *
import threading
import copy

def runAg(agent, connection, protocol, known, pattern, absint):
	""" Method to start an interaction"""
	result = agent.interact(connection, protocol, known, pattern, absint)
	if verbose:
		print "outcome {}".format(result)
	return

def start_interaction(agent1, agent2, prot1, prot2, k_1, k_2, pattern):
	""" Starts interaction between two agents"""
	first_conn, second_conn = Pipe()
	result_1 = []
	result_2 = []

	# this is the abstract interaction
	absint = []
	# agents are two threads connected through a Pipe
	a1 = threading.Thread(target=runAg, args=(agent1, first_conn, prot1, k_1, pattern, absint))
	a2 = threading.Thread(target=runAg, args=(agent2, second_conn, prot2, k_2, pattern, absint))

	a1.start()
	a2.start()
	a1.join()
	a2.join()

	return absint

	# a1.terminate()
	# a2.terminate()


class Protocol():
	""" A protocol is composed of a set of labels ids divided into st, co, nc, re
		And two dictionaries: Dependencies and Text (the NL labels)
	"""

	def __init__(self, labels, text, deps, language):
		self.labels = labels
		self.dependencies = deps
		self.text = text
		self.lang = language

	def get_labels(self):
		return self.labels[0]+self.labels[1]+self.labels[2]+self.labels[3]

	def get_requirements(self):
		return self.labels[1]+self.labels[2]+self.labels[3]

	def is_possible(self, label, interaction):
		""" A label is possible if it has not been said 
			and its requirements are complete
		"""
		if not label in self.get_labels():
			return False			
		dependencies = self.dependencies[label]
		return not [d for d in dependencies if not d in interaction]

	def get_possibilities(self, interaction, restrict):
		return [l for l in self.get_labels() if not l in interaction and self.is_possible(l, interaction)]

	def get_vocabulary(self):
		labels = self.get_labels()
		print "labels {}".format(labels)
		voc = []
		for l in labels:
			print l
			print self.text[l]
			
			text = self.text[l]

			for w in text:
				if w not in voc:
					voc.append(w)
		print "vocabhere : {}".format(voc)
		return voc


class Agent():
	""" A basic agent"""
	def __init__(self, id):
		self.id = id
		self.alignment = {}
		self.interloc = 1-id
		self.mappings_made = {}
		self.just_failed = False
		self.frequency_map = {}
		self.own_frequency = {}

	def __str__(self):
		return str(self.id)

	def __repr__(self):
		return str(self.id)

	def update_own_freq(self, protocol):
		for label in protocol.get_labels():
			t = protocol.text[label]
			for w in t:
				if w in self.own_frequency:
					self.own_frequency[w] += 1
				else:
					self.own_frequency[w] = 1

	def interact(self, connection, protocol, known, pattern, absint):
		"""Start an interaction with an agent"""
		self.update_own_freq(protocol)
		interaction = []
		unknown = [l for l in protocol.get_labels() if not l in known]
		bound = len(pattern)
		self.mappings_made = {}
		for t in pattern: 
			if t==self.id:
				if verbose:
					print ""
					print "I am {} and i am a sender with just failed {}".format(self.id, self.just_failed)
				utterance = 'none'
				label = self.choose_utterance(protocol, known, interaction)
				if not label:
					connection.send('failed')
					if verbose:
						print "failed by sender"
					if self.just_failed:
						# self.punish()
						return 0
					else:
						self.just_failed = True
						continue
				else:
					self.just_failed = False
					utterance = protocol.text[label]
					interaction.append(label)
					absint.append((self.id,label))

					connection.send(utterance)
					if verbose:
						print "Agent {} says {}".format(self.id, utterance)
					
					conf = connection.recv()
					if conf == 'failed':
						if self.just_failed:
							# self.punish()
							return 0
						else:
							self.just_failed = True
							continue
			else:
				received = connection.recv()
				if verbose:
					print ""
					print "I am {} and i received {} with just failed {}".format(self.id, received, self.just_failed)
				if received == 'failed' or received == 'none': 
					if self.just_failed:
						# self.punish()
						return 0
					else:
						self.just_failed = True
						continue

				else:
					interpretation = self.choose_interpretation(protocol, unknown, interaction, received)
					if interpretation == None or interpretation == 0:
						if verbose:
							print "Failed to interpret"
						connection.send('failed')
	
						if self.just_failed:
						# self.punish()
							return 0
						else:
							self.just_failed = True
							continue
	
					else:
						self.just_failed = False
						if verbose:
							print "Agent {} interprets {}".format(self.id, interpretation)
						self.mappings_made[tuple(received)]= protocol.text[interpretation]

						interaction.append(interpretation)
						if verbose:
							print "interaction: {}".format(interaction)
						connection.send('ok')
		return 2

	def reward(self):
		for k in self.mappings_made.keys():
			v = self.mappings_made[k]

			for w in k:
				for ww in v:
					self.alignment[w][ww] += 0.1

	def punish(self):
		for k in self.mappings_made.keys():
			v = self.mappings_made[k]

			for w in k:
				for ww in v:
					self.alignment[w][ww] -= 0.01


	def choose_utterance(self, protocol, known, interaction):
		poss = protocol.get_possibilities(interaction, known)
		if poss: 
			# return the text
			return random.choice(poss)
		return None


	def get_comb_values(self, words, received):

		"""Computes the mapping degree of two sentences"""

		if len(words)<len(received):
			short = words
			long = received

		else:
			short = received
			long = words

		comb_values = {}

		for p in itertools.permutations(list(long), len(short)):

			value = 0
			for i in range(len(short)):
				if len(words)<len(received):
					local = short[i]
					foreign = p[i]
				else:
					local = p[i]
					foreign = short[i]
				if foreign in self.alignment.keys():
					if local in self.alignment[foreign].keys():
						value += self.alignment[foreign][local]/((float(self.own_frequency[local])+float(self.frequency_map[foreign])))
					else:
						value += 0
			if not short:
				comb_values[p] = 0
			else:
				comb_values[p] = (value) - 0.01 * abs(len(received)-len(words))

		val = False
		if len(words)<len(received):
			val = True

		return comb_values, val


	def normalize(self):
		for rec in self.alignment.keys():
			sumV = sum(self.alignment[rec].values())
			if not sumV==0:
				for k in self.alignment[rec].keys():
					self.alignment[rec][k] = self.alignment[rec][k] / sumV


	def comb_update(self, protocol, received, possibilities):
		""" Updates values for possible mappings and retrieves max"""

		values = {}

		upd = {}
		for w in received:
			if not w in self.frequency_map.keys():
				self.frequency_map[w] = 1
			else:
				self.frequency_map[w] += 1

		for pos in possibilities:
			values[pos] = (-20, None)
			# updated = []
			comb_values, val = self.get_comb_values(protocol.text[pos], received)

			for c in comb_values.keys():
				# c is always the LONG one
				value = comb_values[c]

				if value > values[pos][0]:
					values[pos] = (value, c)

				if val:
					foreign = c
					local = protocol.text[pos]
				else:
					foreign = received
					local = c

				for i in range(len(foreign)):
					update = value + 1
					if not foreign[i] in upd.keys() or not local[i] in upd[foreign[i]].keys():
						if not foreign[i] in upd.keys():
						# 	upd[foreign[i]] = {}
							upd[foreign[i]] = {}
						# if not local[i] in upd[foreign[i]].keys():
						# 	upd[foreign[i]][local[i]] = [0,0]

						upd[foreign[i]][local[i]] = update
						
							
						# upd[foreign[i]][local[i]][0] += update
						# upd[foreign[i]][local[i]][1] +=1
					else:
						if update>upd[foreign[i]][local[i]]:
							upd[foreign[i]][local[i]] = update

		for k in upd.keys():
			if not k in self.alignment.keys():
				self.alignment[k] = {}
			for kk in upd[k].keys():
				if kk in self.alignment[k].keys() and self.alignment[k][kk]>0:

					self.alignment[k][kk] += upd[k][kk]
					# self.alignment[k][kk] += upd[k][kk][0]/upd[k][kk][1]
					# self.alignment[k][kk] += (upd[k][kk])/self.alignment[k][kk]
				else:
					self.alignment[k][kk] = upd[k][kk]
					# self.alignment[k][kk] = upd[k][kk][0]/upd[k][kk][1]

		# self.normalize()
						# updated.append((foreign[i],local[i]))
		return values


	def choose_interpretation(self, protocol, restrict, interaction, received):
		""" Choose the interpretation for a message and perform the updates """

		#received is a set of words
		for i in [0]:
			for w in received:
				if not w in self.alignment.keys():
					self.alignment[w] = {}

		possibilities = protocol.get_possibilities(interaction, restrict)

		if not possibilities:
			return 0
		values = self.comb_update(protocol, received, possibilities)
		chosen = max(possibilities, key=lambda x : values[x][0])

		if verbose:
			print "-------------------"
			print "received {}".format(received)
			print possibilities
			print "interpretation possibilities"
			for p in possibilities:
				print (p, protocol.text[p], values[p])

		if verbose and not chosen==None:
			print "interpretation chosen: {}".format(protocol.text[chosen])
			
		return chosen


def isSuccess(absint, alg_st, prot_es):
	"""Returns True if the interaction is successful:
		- all labels are said
		- all dependencies are ok
	"""

	esint = []

	for i in absint:
		if i[0]==1:
			esint.extend([l for l in alg_st[i[1]] if unicode(l) in prot_es.get_labels()])
		else:
			esint.append(i[1])

	suc = (not [u for u in prot_es.get_labels() if str(u) not in esint])
	if verbose:
		print "alg {}".format(alg_st)
		print "labels es {}".format(prot_es.get_labels())
		print "esint {}".format(esint)
		print "suc 1 : {}".format(suc)
	for i in range(len(esint)):
		suc = suc and prot_es.is_possible(esint[i], esint[:i])
	
	return suc




############################## PROTOCOL BUILDING #####################################

def build_protocol(ltext, ddep):
	""" Build protocol from text and dependencies"""
	protocol = {}

	labels = ([],[],[],[])
	for l in ltext:
		lang = l[0]
		if l[1] == 'st':
			labels[0].append(l[2])
		elif l[1] == 'co':
			labels[1].append(l[2])
		elif l[1] == 'nc':
			labels[2].append(l[2])		
		elif l[1] == 're':
			labels[3].append(l[2])

	text = {}
	for l in ltext:
		# nouns = [w for w in l[3][0][1] if len(w)>2]
		nouns = [w for w in l[3][0][1]]
		text[l[2]] = l[3][0][0]+nouns[: min(6, len(nouns))]

	dependencies = {}
	for l in ltext:
		if l[2] in ddep.keys():
			dependencies[l[2]] = ddep[l[2]]
		else:
			dependencies[l[2]] = []

	protocol = Protocol(labels, text, dependencies, lang)

	return protocol

def build_protocol_fl(ltext, ddep):
	""" Build protocol from a freeling text and dependencies"""
	protocol = {}
	labels = ([],[],[],[])
	for l in ltext:
		lang = l[0]
		if l[1] == 'st':
			labels[0].append(l[2])
		elif l[1] == 'co':
			labels[1].append(l[2])
		elif l[1] == 'nc':
			labels[2].append(l[2])		
		elif l[1] == 're':
			labels[3].append(l[2])

	text = {}
	for l in ltext:
		words = [w[1] for w in l[3][0]] # this takes just the first sentence for each protocol
		text[l[2]] = words[: min(5, len(words))]

	dependencies = {}
	for l in ltext:
		if l[2] in ddep.keys():
			dependencies[l[2]] = ddep[l[2]]
		else:
			dependencies[l[2]] = []

	protocol = Protocol(labels, text, dependencies, lang)

	return protocol

def read_dependencies():
	"""Parse the dependencies file"""
	with open('oxford_translation/data/all_dependencies_de_duplicated.txt') as f:
	# with open('all_labels.txt's) as f:
		content = f.readlines()
	# remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content]
	dependencies = []
	iprot = -1
	jpair = 1
	labels = []
	for l in content:
		# if iprot>=1:
		#     break
		if not l: 
			continue
		elif l.startswith("#PROTOCOLPAIR"):    
			iprot += 1
			dependencies.append([{},{}])
		elif l.startswith('#PAIR'):
			jpair = 1 - jpair
		elif not l.startswith("#"):
			dep, lab = l.split()
			if lab in dependencies[iprot][jpair]:
				dependencies[iprot][jpair][lab].append(dep)
			else:
				dependencies[iprot][jpair][lab] = [dep]
	return dependencies




############################## ALIGNMENT METHODS #####################################

def read_alignment_req():
	"""Parse the alignment file and return an english-spanish alignment"""
	with open('oxford_translation/results.txt') as f:
		content = f.readlines()
		f.close()
	# remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content]
	alignment = {}
	for l in content:
		if l:
			es, en, conf = l.split()
			if not en in alignment.keys():	
				alignment[unicode(en)] = [unicode(es)]
			else:
				alignment[unicode(en)].append(unicode(es))
	return alignment


def find_alignment_st(prot_en, prot_es):
	"""Method to find the true alignment 
		between the st labels of two protocols
	"""

	en_dep = prot_en.dependencies
	es_dep = prot_es.dependencies

	relevant_en = []
	relevant_es = []

	for lan in ((prot_en, relevant_en),(prot_es, relevant_es)):
		prot = lan[0]
		relevant = lan[1]
		for d1 in prot.dependencies.keys():
			if d1 in prot.labels[0]:
				for d2 in prot.dependencies[d1]:
					if d2 in prot.labels[0]:
						relevant.append([d2,d1])
	sorted_es = []
	sorted_en = []

	for lan in ((relevant_en, sorted_en), (relevant_es, sorted_es)):
		relevant = lan[0]
		sorted = lan[1]
		first = [p for p in relevant if not [q for q in relevant if p[0]==q[1]]][0]
		last = [p for p in relevant if not [q for q in relevant if p[1]==q[0]]][0][1]
		sorted.extend(first)
		while not sorted[-1]==last:
			next = [p for p in relevant if sorted[-1]==p[0]][0]
			sorted.append(next[1])

	alg = {sorted_en[i] : sorted_es[i] for i in range(len(sorted_en))}
	return alg

def best_maps_dict(alignment):
	"""Gets only the mappings with best value, as a dictionary"""
	res = {}
	for k in alignment.keys():
		max_keys = []
		highest = None
		if alignment[k]:
			highest = max(alignment[k].values())
			max_keys = [kk for kk in alignment[k].keys() if alignment[k][kk]==highest]

		res[k] = max_keys

	return res

def good_maps(alignment):
	"""Gets only the mappings with value>0"""
	res = []
	for k in alignment.keys():
		good = []
		if alignment[k]:
			good = [(w, alignment[k][w])  for w in alignment[k].keys() if alignment[k][w]>0]

		res.append((k, good))

	return res


def precision_recall(alignment,  reference):
	if not alignment: 
		return 0,0
	else:
		max_alg = {k : max(alignment[k].iteritems(), key=itemgetter(1))[0] for k in alignment.keys()}
		correct = sum(1 for k in alignment.keys() if max_alg[k] == reference[k])
		return (float(correct)/float(len(alignment.keys())), float(correct)/float(len(reference.keys())))


def reverseAlg(alignment):
	return {alignment[k] : k for k in alignment.keys()}


############################## EXPERIMENT #####################################

def experiment(inters, reward, punish, name):

	# get the text
	with open('data/clean-labels-freeling-dedup.json') as data_file:    
		text = json.load(data_file)
		data_file.close()

	# get the dependencies
	dependencies = read_dependencies()

	# create agents
	a_es = Agent(0)
	a_en = Agent(1)

	successes = []
	protocols = [] # we need to remember the used protocols for the eval
	alg_req = read_alignment_req()
	for h in range(inters):
		# print h
		if verbose:
			print "interaction {}".format(h)

		forbidden = [218]
		choices = [i for i in range(327) if not i in forbidden]
		prot = random.choice(choices)
		if not prot in protocols:
			protocols.append(prot)
		if verbose:
			print "Protocol {}".format(prot)

		# Build the protocols
		prot_en = build_protocol_fl(text[prot][0], dependencies[prot][0])
		prot_es = build_protocol_fl(text[prot][1], dependencies[prot][1])

		# Build the labels alignment
		alg_st = find_alignment_st(prot_en, prot_es) # steps alignment
		# requirements alignment
		alg = {k : [alg_st[k]] for k in alg_st}
		for l in prot_en.get_requirements():	
			alg[l] = alg_req[l]

		# Divide known labels for each agent
		k_en = random.sample(prot_en.get_labels(), len(prot_en.get_labels())/2)
		k_es = [alg[w] for w in prot_en.get_labels() if not w in k_en]

		# Build the interaction patterns
		patterns = [[0,1] for p in range(len(prot_en.get_labels()))]
		pattern = [e for l in patterns for e in l]

		#Start the interaction
		absint = start_interaction(a_en,a_es, prot_en, prot_es, k_en, k_es, pattern)

		if isSuccess(absint, alg, prot_es):
			if verbose:
				print "success"
			successes.append(1)
			if reward:
				a_en.reward()
				a_es.reward()
		else:
			if verbose:
				print "not success"
			successes.append(0)	
			if punish:
				a_en.punish()
				a_es.punish()

	with open('results/{}.txt'.format(name), 'w+') as res_file:
		res_file.write("Alignment en \n")
		# res_file.write(repr(best_maps(a_en.alignment)))
		res_file.write(repr(good_maps(a_en.alignment)))
		res_file.write("\n \n ------------------------------------------------------------ \n \n")
		res_file.write("Alignment es \n")
		# res_file.write(repr(best_maps(a_es.alignment)))
		res_file.write(repr(good_maps(a_es.alignment)))
	if verbose:
		print len(a_en.alignment)
		print len(a_es.alignment)

	print "-------------------"	
	print "successes rate: {}".format(sum(successes))
	# print "successes: {}".format(successes)
	print "-------------------"
	print "English Agent"
	pres, rec = evaluate(best_maps_dict(a_en.alignment), 'en', protocols)
	# print "Precision: {}".format(get_precision_oxford(best_maps(a_en.alignment), 'en'))
	# print "Mod Precision: {}".format(get_mod_precision_oxford(best_maps(a_en.alignment), 'en'))
	print "-------------------"
	print "Spanish Agent"
	# print "Precision: {}".format(get_precision_oxford(best_maps(a_es.alignment), 'es'))
	# print "Mod Precision: {}".format(get_mod_precision_oxford(best_maps(a_es.alignment), 'es'))
	pres, rec = evaluate(best_maps_dict(a_es.alignment), 'es', protocols)

	return 


def main(argv):

	name = 'test'
	inters = 300
	reward = 0
	punish = 0

	try:
		opts, args = getopt.getopt(argv,"i:f:",["interactions=", "feedback="])
	except getopt.GetoptError:
		print '-a agent type \n -i number of interactions \n -s protocol size'
		sys.exit(2)
	for opt, arg in opts:
		if not arg=='':
			if opt == '-h':
				print '-a agent type \n -i number of interactions \n -s protocol size'
				sys.exit()

			if opt in ("-i", "--interactions"):
				inters = int(arg)
			if opt in ("-f", "--feedback"):
				if opt == 'p':
					punish = 1
				elif opt == 'r':
					reward = 1
				elif opt=='rp' or opt=='pr':
					reward = 1
					punish = 1

	for i in range(1):
		experiment(inters, reward, punish, 'test')


global verbose 
verbose = 1

if __name__ == "__main__":
   main(sys.argv[1:])


