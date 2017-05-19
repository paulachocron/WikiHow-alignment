
#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, getopt
from multiprocessing import Process, Pipe, Queue
import random
#from scipy.stats import rv_discrete

from operator import itemgetter
import numpy as np
import itertools
import json
import time
from evaluator import get_precision
import threading

def runAg(agent, connection, protocol, known, pattern):
	result = agent.interact(connection, protocol, known, pattern)
	if verbose:
		print "outcome {}".format(result)
	return

def start_interaction(agent1, agent2, prot1, prot2, k_1, k_2, pattern):
	""" Starts interaction between two agents"""
	first_conn, second_conn = Pipe()
	result_1 = []
	result_2 = []

	# agents are two threads connected through a Pipe
	a1 = threading.Thread(target=runAg, args=(agent1, first_conn, prot1, k_1, pattern))
	a2 = threading.Thread(target=runAg, args=(agent2, second_conn, prot2, k_2, pattern))

	a1.start()
	a2.start()
	a1.join()
	a2.join()			
	# a1.terminate()
	# a2.terminate()


class Protocol():
	""" A protocol is composed of a set of labels ids divided into st, co, nc, re
		And two dictionaries: Dependencies and Text (the NL labels)
	"""

	def __init__(self, labels, text, deps):
		self.labels = labels
		self.dependencies = deps
		self.text = text

	def get_labels(self):
		return self.labels[0]+self.labels[1]+self.labels[2]+self.labels[3]


	def is_possible(self, label, interaction):
		""" A label is possible if it has not been said 
			and its requirements are complete
		"""
		if not label in self.get_labels():
			return False			
		dependencies = self.dependencies[label]
		return not [d for d in dependencies if not d in interaction]

	def get_possibilities(self, interaction, restrict):
		return [l for l in restrict if not l in interaction and self.is_possible(l, interaction)]


class Agent():
	""" A basic agent"""
	def __init__(self, id):
		self.id = id
		self.alignment = {}
		self.interloc = 1-id

	def __str__(self):
		return str(self.id)

	def __repr__(self):
		return str(self.id)

	def interact(self, connection, protocol, known, pattern):
		"""Start an interaction with an agent"""
		interaction = []
		unknown = [l for l in protocol.get_labels() if not l in known]
		bound = len(pattern)
		mappings_made = {}
		for t in pattern: 
			if t==self.id:
				if verbose:
					print ""
					print "I am {} and i am a sender".format(self.id)
				utterance = 'none'
				label = self.choose_utterance(protocol, known, interaction)
				if not label:
					interaction.append(None)
					if interaction and interaction[-1] == None:
						connection.send('failed')
						if verbose:
							print "failed by sender"
						return 0
				else:
					utterance = protocol.text[label]
					interaction.append(label)

				connection.send(utterance)
				if verbose:
					print "Agent {} says {}".format(self.id, utterance)
				
				conf = connection.recv()
				if conf == 'failed':
					return 0
			else:
				received = connection.recv()
				if verbose:
					print ""
					print "I am {} and i received {}".format(self.id, received)
				if received == 'failed':
					return 0
				
				if received == 'none':
					interpretation = None
				else:
					interpretation = self.choose_interpretation(protocol, unknown, interaction, received)
					if interpretation == None:
						if verbose:
							print "Failed to interpret"
						connection.send('failed')
						return 0	
					if verbose:
						print "Agent {} interprets {}".format(self.id, interpretation)
					if interpretation == 0:
						# print interaction
						connection.send('failed')
						if verbose:
							print "failed by receiver"
						return 0
				interaction.append(interpretation)
				if verbose:
					print "interaction: {}".format(interaction)
				connection.send('ok')
		return 2


	def choose_utterance(self, protocol, known, interaction):
		poss = protocol.get_possibilities(interaction, known)
		if poss: 
			# return the text
			return random.choice(poss)
		return None
	
	def best_comb(self, words, received):
		"""Computes the mapping degree of two sentences"""
		# this uses permutations to select the one with best value
		# it still could be better by using the size of the sentences

		if len(words)<len(received):
			short = words
			long = received

		else:
			short = received
			long = words

		# if words and received:
		# 	total = 0
		# else:
		# 	total = 0

		# for w in received:
		# 	best = None
		# 	max = 0
		# 	for v in words:
		# 		if w in self.alignment and words[0] in self.alignment[w]:
		# 			if self.alignment[w][words[0]]>max:
		# 				max = self.alignment[w][words[0]]
		# 	total += max

		# return total

		max = 0
		for p in itertools.permutations(list(long), len(short)):
			value = 0
			for i in range(len(short)):
				if len(words)<len(received):
					local = short[i]
					foreign = p[i]
				else:
					local = p[i]
					foreign = short[i]
				if foreign in self.alignment.keys() and local in self.alignment[foreign].keys():
					value += self.alignment[foreign][local]
			if value >= max:
				max = value
				best = p
		if max<=0:
			best = None

		val = False
		if len(words)<len(received):
			val = True

		return max, best, val


	def best_match(self, protocol, possibilities, received):
		"""Get the mapping with greatest value for a received sentence 
			It returns the id of the best map, the 
		"""
		if not possibilities:
			return None, {}, {}

		values = {pos: self.best_comb(protocol.text[pos], received) for pos in possibilities}

		best = max(possibilities, key=lambda x : values[x])
		# sorted = sort(possibilities, key=lambda x : self.mapping_value(x, received))
		sortedIn = sorted(list(set([values[x][0] for x in possibilities])))

		indval = {pos: sortedIn.index(values[pos][0]) for pos in possibilities}

		return best, indval, values



	def choose_interpretation(self, protocol, restrict, interaction, received):
		"""Critical method!! This chooses the interpretation for a message
			And performs the updates
		"""
		#received is a set of words
		for i in [0]:
			for w in received:
				if not w in self.alignment.keys():
					self.alignment[w] = {}

		possibilities = protocol.get_possibilities(interaction, restrict)
		chosen, indval, values = self.best_match(protocol, possibilities, received)

		if verbose:
			print "received {}".format(received)
			print possibilities
			print "interpretation possibilities {}".format([(p, protocol.text[p], values[p]) for p in possibilities])
			
		# update values with rewards
		for i in [1]:
			for jr in range(len(received)):
				w = received[jr]
				for pos in possibilities:
					ind = indval[pos]
					posT = protocol.text[pos]
					# for jo in range(len(posT[i])):
					for jo in range(len(posT)):
						# wo = posT[i][jo]
						wo = posT[jo]
						absV = abs(jo-jr)
						if wo in self.alignment[w]:
							# self.alignment[w][wo] += 1
							#self.alignment[w][wo] += (1.0+self.mapping_value(posT, received))/(absV+1)
							# self.alignment[w][wo] += 1.0 + (1.0*(ind+1))
							self.alignment[w][wo] += (1.0*(ind+1))/(absV+1)
							# self.alignment[w][wo] += (1.0)/(absV+1)
						else:
							# self.alignment[w][wo] = 1
							# self.alignment[w][wo] = (1.0*(ind+1))
							self.alignment[w][wo] = (1.0*(ind+1))/(absV+1)
							# self.alignment[w][wo] = 1.0/(absV+1)

							#self.alignment[w][wo] = (1.0+self.mapping_value(posT, received))/(absV+1)
						if verbose:
							print "mapping {} {} + {}".format(w,wo,(1.0*(ind+1))/(absV+1))

		# update with punishments
		for w in received:
			for wo in self.alignment[w]:
				# if not [1 for pos in possibilities if wo in protocol.text[pos][0] or wo in protocol.text[pos][1]]:
				if not [1 for pos in possibilities if wo in protocol.text[pos]]:
					self.alignment[w][wo] -= 1
					if verbose:
						print "mapping {} {} - {}".format(w,wo,1)

		if verbose:
			print "alignment"
			print self.alignment 
			
		# and now choose one

		return chosen


############################## PROTOCOL BUILDING #####################################

def build_protocol(ltext, ddep):
	""" Build protocol from text and dependencies"""
	protocol = {}

	labels = ([],[],[],[])
	for l in ltext:
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
		nouns = [w for w in l[3][0][1] if len(w)>2]
		text[l[2]] = l[3][0][0]+nouns[: min(4, len(nouns))]

	dependencies = {}
	for l in ltext:
		if l[2] in ddep.keys():
			dependencies[l[2]] = ddep[l[2]]
		else:
			dependencies[l[2]] = []

	protocol = Protocol(labels, text, dependencies)

	return protocol

def build_protocol_fl(ltext, ddep):
	""" Build protocol from a freeling text and dependencies"""
	protocol = {}
	labels = ([],[],[],[])
	for l in ltext:
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
		words = [w[1] for w in l[3][0]]
		text[l[2]] = words[: min(5, len(words))]

	dependencies = {}
	for l in ltext:
		if l[2] in ddep.keys():
			dependencies[l[2]] = ddep[l[2]]
		else:
			dependencies[l[2]] = []

	protocol = Protocol(labels, text, dependencies)

	return protocol

def read_dependencies():
	"""Parse the dependencies file"""
	with open('data/all_dependencies.txt') as f:
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

def find_alignment(prot_en, prot_es):
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

	alg = [(sorted_en[i],sorted_es[i]) for i in range(len(sorted_en))]
	return alg


def best_maps(alignment):
	"""Gets only the mappings with best value"""
	res = []
	for k in alignment.keys():
		max_keys = []
		highest = None
		if alignment[k]:
			highest = max(alignment[k].values())
			max_keys = [kk for kk in alignment[k].keys() if alignment[k][kk]==highest]

		res.append((k, max_keys, highest))

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

def experiment(inters, text, dependencies, name):
	# create agents
	a_es = Agent(0)
	a_en = Agent(1)


	for h in range(inters):
		# choose a protocol 
		prot = random.choice(range(650))
		# print prot
		# prot = 500
		if verbose:
			print "Protocol {}".format(prot)
		# build protocols
		prot_en = build_protocol_fl(text[prot][0], dependencies[prot][0])
		prot_es = build_protocol_fl(text[prot][1], dependencies[prot][1])

		alg = find_alignment(prot_en, prot_es)

		k_en = random.sample(prot_en.labels[0], len(prot_en.labels[0])/2)
		k_es = [[p[1] for p in alg if p[0]==k][0] for k in prot_en.labels[0] if not k in k_en]

		if prot_en.labels[3]==[] and prot_es.labels[3]==[]:
			w = random.choice([0,1])
			k_en.extend(prot_en.labels[w+1])
			k_es.extend(prot_es.labels[(1-w)+1])

		else:
			w = random.choice(['en', 'es'])
			if w=='en':
				k_en.extend(prot_en.labels[1]+prot_en.labels[2]+prot_en.labels[3])
			else:
				k_es.extend(prot_es.labels[1]+prot_es.labels[2]+prot_es.labels[3])

		patterns = [[0,1] for p in range((len(prot_en.labels[0])/2)+4)]
		pattern = [e for l in patterns for e in l]

		start_interaction(a_en,a_es, prot_en, prot_es, k_en, k_es, pattern)

	with open('results/{}.txt'.format(name), 'w+') as res_file:
		res_file.write("Alignment en \n")
		res_file.write(repr(best_maps(a_en.alignment)))
		res_file.write("\n \n ------------------------------------------------------------ \n \n")
		res_file.write("Alignment es \n")
		res_file.write(repr(best_maps(a_es.alignment)))
	print len(a_en.alignment)
	print len(a_es.alignment)

	# print "Alignment en"
	# print len(best_maps(a_en.alignment))
	# print good_maps(a_en.alignment)
# 	print best_maps(a_en.alignment)
# # 
# 	# print a_en.alignment
# 	print ""
# 	# print "Precision: {}".format(get_precision(best_maps(a_en.alignment), 'en'))
# 	print ""
# 	print "Alignment es"
# 	print best_maps(a_es.alignment)
# 	# print good_maps(a_es.alignment)
# 	# print a_es.alignment
# 	print ""
# 	# print "Precision: {}".format(get_precision(best_maps(a_es.alignment), 'es'))
# 	print ""
# 	print "------------------------------"
# 	print ""

	return


def main(argv):

	name = 'test'
	inters = 2000
	ver = 0

	try:
		opts, args = getopt.getopt(argv,"a:i:s:v:p:",["agent=","interactions=","size=", "verbose=", "protocol="])
	except getopt.GetoptError:
		print '-a agent type \n -i number of interactions \n -s protocol size'
		sys.exit(2)
	for opt, arg in opts:
		if not arg=='':
			if opt == '-h':
				print '-a agent type \n -i number of interactions \n -s protocol size'
				sys.exit()
			if opt in ("-a", "--agent"):
				try:	
					agentType = int(arg)
				except NameError:
					print "-a must be 0,1,2, or 3"
			if opt in ("-i", "--interactions"):
				inters = int(arg)
			if opt in ("-s", "--size"):
				size = int(arg) 
			if opt in ("-v", "--verbose"):
				ver = int(arg) 
			if opt in ("-p", "--protocol"):
				prot = arg

	# read the text as json
	with open('data/clean-labels-freeling.json') as data_file:    
		labels = json.load(data_file)

	experiment(inters, labels, read_dependencies(), 'test')


global verbose 
verbose = 0

if __name__ == "__main__":
   main(sys.argv[1:])


