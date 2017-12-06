#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, getopt
from multiprocessing import Pipe
import random
from operator import itemgetter
import itertools
import json
import threading
import copy
from protocols import *

def runAg(agent, connection, protocol, known, pattern, absint, learn):
	""" Method to start an interaction"""
	result = agent.interact(connection, protocol, known, pattern, absint, learn)
	if verbose:
		print "outcome {}".format(result)
	return

def start_interaction(agent1, agent2, prot1, prot2, k_1, k_2, pattern, learn=1):
	""" Starts interaction between two agents"""
	first_conn, second_conn = Pipe()
	result_1 = []
	result_2 = []

	# this is the abstract interaction
	absint = []
	# agents are two threads connected through a Pipe
	a1 = threading.Thread(target=runAg, args=(agent1, first_conn, prot1, k_1, pattern, absint, learn))
	a2 = threading.Thread(target=runAg, args=(agent2, second_conn, prot2, k_2, pattern, absint, learn))

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

	def __init__(self, id, labels, text, deps, language):
		self.id = id
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
		self.played = []

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

	def interact(self, connection, protocol, known, pattern, absint, learn):
		"""Start an interaction with an agent"""
		if not protocol.id in self.played:
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
						print "Agent {} says {} : {}".format(self.id, label, utterance)
					
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
					interpretation = self.choose_interpretation(protocol, unknown, interaction, received, learn)
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
		self.played.append(protocol.id)
		return 2

	def assign_alignment(self, alg):
		self.alignment = alg


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

		# if verbose:
		# 	print ""
		# 	print "Comb values"

		for p in itertools.permutations(list(long), len(short)):

			value = 0
			for i in range(len(short)):
				if len(words)<len(received):
					local = short[i]
					foreign = p[i]
				else:
					local = p[i]
					foreign = short[i]
				if foreign in self.alignment:
					if local in self.alignment[foreign].keys():
						# print local
						# print foreign
						# print 1
						# print self.alignment[foreign][local]
						# print 2
						# print self.own_frequency[local]
						# print 3
						# print foreign
						# print self.frequency_map
						# print self.frequency_map[foreign]
						dist = abs(i-long.index(p[i]))
						if dist>2:
							tm = dist-2
						else:
							tm = 0
						# value += self.alignment[foreign][local]/((float(self.own_frequency[local])+float(self.frequency_map[foreign]))+abs(i-long.index(p[i])))
						# value += self.alignment[foreign][local]/((float(self.own_frequency[local])+float(self.frequency_map[foreign]))+tm)
						value += self.alignment[foreign][local]/((float(self.own_frequency[local])+float(self.frequency_map[foreign])))	
						# value += self.alignment[foreign][local]/((float(self.own_frequency[local])+float(self.frequency_map[foreign]))) - tm/1000
						# value += self.alignment[foreign][local]/100
					else:
						value += 0
			if not short:
				comb_values[p] = 0
			else:
				comb_values[p] = value - 0.005 * abs(len(received)-len(words))

			# if verbose:
			# 	print "{}: {}  {}".format(long, p, comb_values[p])

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


	def comb_update(self, protocol, received, possibilities, learn):
		""" Updates values for possible mappings and retrieves max"""

		values = {}

		upd = {}
		if 1:
			for w in received:
				if not protocol.id in self.played:

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
					update = value + 0.1
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
		if learn:
			for k in upd.keys():
				if not k in self.alignment:
					self.alignment[k] = {}
				for kk in upd[k].keys():
					if kk in self.alignment[k].keys() and self.alignment[k][kk]>0:

						self.alignment[k][kk] += upd[k][kk]
						# self.alignment[k][kk] += upd[k][kk][0]/upd[k][kk][1]
						# self.alignment[k][kk] += (upd[k][kk])/self.alignment[k][kk]
					else:
						self.alignment[k][kk] = upd[k][kk]
						# self.alignment[k][kk] = upd[k][kk][0]/upd[k][kk][1]

		return values

	def choose_interpretation(self, protocol, restrict, interaction, received, learn):
		""" Choose the interpretation for a message and perform the updates """

		#received is a set of words
		for i in [0]:
			for w in received:
				if not w in self.alignment:
					self.alignment[w] = {}

		possibilities = protocol.get_possibilities(interaction, restrict)

		if not possibilities:
			return 0
		values = self.comb_update(protocol, received, possibilities, learn)
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
		print "diffs 1: {}".format([i for i in prot_es.get_labels() if not i in esint])
		print "diffs 2: {}".format([i for i in esint if not i in prot_es.get_labels()])
	for i in range(len(esint)):
		suc = suc and prot_es.is_possible(esint[i], esint[:i])
	
	return suc




############################## PROTOCOL BUILDING #####################################

def build_protocol_fl(prot, ltext, ddep, bound):
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
		text[l[2]] = words[: min(bound, len(words))]

	dependencies = {}
	for l in ltext:
		if l[2] in ddep.keys():

			dependencies[l[2]] = ddep[l[2]]
		else:
			dependencies[l[2]] = []

	protocol = Protocol(prot, labels, text, dependencies, lang)

	return protocol

def read_dependencies_dict():
	with open('data/dependencies.json') as data_file:    
		deps = json.load(data_file)
		data_file.close()
	return deps


############################## ALIGNMENT METHODS #####################################

def read_alignment_req():
	"""Parse the alignment file and return an english-spanish alignment"""
	with open('data/semantic_alg.txt') as f:
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

	# print alignment
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

############################## EXPERIMENT #####################################

def get_en_trans(alg, l):
	trans = [k for k in alg.keys() if l in alg[k]]
	if trans:
		return trans[0]
	return False


def interact(learning, alg_req, prot_en, prot_es, a_es, a_en):
	if verbose:
		print "Protocol {}".format(prot)
		if learning:
			print "interaction {}".format(h)
		else:
			print "test interaction {}".format(h)
		
	# Build the protocols

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
	absint = start_interaction(a_en,a_es, prot_en, prot_es, k_en, k_es, pattern,learn=learning)

	if isSuccess(absint, alg, prot_es):
		if verbose:
			print "success"
		return 1
	else:
		if verbose:
			print "not success"
		return 0


def experiment(inters, test, protocols, test_prot, name):
	# get the text
	with open('data/clean-labels-freeling-dedup.json') as data_file:    
		text = json.load(data_file)
		data_file.close()

	# get the dependencies
	dependencies = read_dependencies_dict()

	leng = 6
	# create agents
	a_es = Agent(0)
	a_en = Agent(1)

	successes = []
	alg_req = read_alignment_req()

	for h in range(inters):
		prot = protocols[h]
		prot_en = build_protocol_fl(prot, text[prot][0], dependencies[prot][0], leng)
		prot_es = build_protocol_fl(prot, text[prot][1], dependencies[prot][1], leng)
		interact(1, alg_req, prot_en, prot_es, a_es, a_en)
		
	# now the test phase
	print ""
	print "TEST PHASE"
	print ""

	successes = []

	for h in range(test):
		prot = test_prot[h]
		prot_en = build_protocol_fl(prot, text[prot][0], dependencies[prot][0], leng)
		prot_es = build_protocol_fl(prot, text[prot][1], dependencies[prot][1], leng)
		success = interact(0, alg_req, prot_en, prot_es, a_es, a_en)
		successes.append(success)

	if test==0:
		succrate = 0
	else:
		succrate = sum(successes)/float(test)

	print "successes rate: {}".format(succrate)

	return succrate


def main(argv):

	name = 'test'
	training = 100
	test = 100	
	i = 1

	try:
		opts, args = getopt.getopt(argv,"t:p:r:",["training=", "precision=", "protocols="])
	except getopt.GetoptError:
		print '-t number of training interactions \n -r protocol set'
		sys.exit(2)
	for opt, arg in opts:
		if not arg=='':
			if opt == '-h':
				print '-t number of training interactions \n -r protocol set'
				sys.exit()
			if opt in ("-t", "--training"):
				training = int(arg)
			if opt in ("-r", "--protocols"):
				if not i in [0,1,2,3,4]:
					print 'The protocol must be between 0 and 4 (inclusive)'
					sys.exit(2)					
					i = int(arg)
	
	protocols = [prots0,prots1,prots2,prots3,prots4]
	test_protocols = [test_prots0,test_prots1,test_prots2,test_prots3,test_prots4]

	res = experiment(training, test, protocols[i], test_protocols[i], 'test')
	

global verbose 
verbose = 0

if __name__ == "__main__":
   main(sys.argv[1:])