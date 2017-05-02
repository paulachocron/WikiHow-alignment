
#!/usr/bin/python

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

# verbose = 0
# if remote:
# 	verbose = 0

def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	return np.exp(x) / np.sum(np.exp(x), axis=0)

def start_interaction(agent1, agent2, prot1, prot2, k_1, k_2, pattern):
	""" Starts interaction between two agents"""
	first_conn, second_conn = Pipe()
	queue = Queue()
	result_1 = []
	result_2 = []
	a1 = Interlocutor(agent1, first_conn, queue, prot1, k_1, pattern)
	a2 = Interlocutor(agent2, second_conn, queue, prot2, k_2, pattern)

	a1.start()
	a2.start()
	a1.join()
	a2.join()

	while not queue.empty():
		history = queue.get()
		if agent1.id == history[0]: 
			agent1.update_alignment(history)
			# agent1.update_alignment(history[1], history[2], history[3], history[4], history[5])
		elif agent2.id == history[0]:
			agent2.update_alignment(history)
			# agent2.update_alignment(history[1], history[2], history[3], history[4], history[5])
			
	a1.terminate()
	a2.terminate()


class Protocol():

	def __init__(self, vocabulary, labels, deps):
		self.vocabulary = vocabulary
		self.dependencies = deps
		self.text = labels

	def get_labels(self):
		return self.vocabulary[0]+self.vocabulary[1]+self.vocabulary[2]+self.vocabulary[3]


class Interlocutor(Process):
	""" An interlocutor that relates a process with an agent"""
	
	def __init__(self, agent, connection, queue, protocol, known, pattern):
		super(Interlocutor, self).__init__()
		self.agent = agent
		self.connection = connection
		self.queue = queue
		self.protocol = protocol
		self.known = known
		self.pattern = pattern

	def run(self):
		result = self.agent.interact(self.connection, self.protocol, self.known, self.pattern)
		if verbose:
			print "outcome {}".format(result)
		# this should also be an agent's method (something like "remember")
		self.queue.put([self.agent.id, self.agent.alignment])
		return

# a protocol is dict : ids -> ((verbs, nouns), [ids]) which mean text, dependencies
# and interaction is [ids]

def is_possible(label, protocol, interaction):
	if not label in protocol.get_labels():
		return False
			
	dependencies = protocol.dependencies[label]
	# print label 
	# print dependencies
	return not [d for d in dependencies if not d in interaction]

def get_possibilities(protocol, interaction, restrict):
	# print "labels {}".format(protocol.get_labels())
	# print "restrict {}".format(restrict)
	# print protocol.dependencies
	return [l for l in restrict if not l in interaction and is_possible(l, protocol, interaction)]


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

	def initialize(self, word):
		"""Get initial alignment"""
		random.shuffle(self.vocabulary)
		self.alignment[word] = {v : (1.0/(len(self.unknown))) for v in self.unknown}
		return

	def best_map(self, foreign):
		maxVal = max(self.alignment[foreign].values())
		if self.alignment[foreign].values().count(maxVal)>1:
			return None
		best = [v for v in self.vocabulary if self.alignment[foreign][v]==maxVal][0]
		return best


	def interact(self, connection, protocol, known, pattern):
		"""Start an interaction with an agent"""
		interaction = []
		unknown = [l for l in protocol.get_labels() if not l in known]
		bound = len(pattern)
		mappings_made = {}
		for t in pattern: 
			if t==self.id:
				if verbose:
					print "I am {} and i am a sender".format(self.id)
				utterance = 'none'
				label = self.choose_utterance(protocol, known, interaction)
				if not label:
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
					print "I am {} and i received {}".format(self.id, received)
				if received == 'failed':
					return 0
	
				interpretation = self.choose_interpretation(protocol, unknown, interaction, received)
				if interpretation == None and received != 'none':
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


	def update_alignment(self, history):
		""" Updates the alignment after an interaction"""
		self.alignment = history[1]

	def choose_utterance(self, protocol, known, interaction):
		poss = get_possibilities(protocol, interaction, known)
		if poss: 
			# return the text
			return random.choice(poss)
		return None
	
	def best_comb(self, words, received):
		# use permutations : )
		# and maybe divide it by the lenght of long?

		if len(words)<len(received):
			short = words
			long = received

		else:
			short = received
			long = words

		best = None
		max = 0
		for p in itertools.permutations(long, len(short)):
			value = 0
			for i in range(len(short)):
				if len(words)<len(received):
					local = short[i]
					foreign = p[i]
				else:
					local = p[i]
					foreign = short[i]
				if foreign in self.alignment and local in self.alignment[foreign]:
					value += self.alignment[foreign][local]
			if value >= max:
				max = value
				best = p

		return max

	# def best_comb(self, words, received):
	# 	# use permutations : )
	# 	# and maybe divide it by the lenght of long?

	# 	if len(words)<len(received):
	# 		short = words
	# 		long = received

	# 	else:
	# 		short = received
	# 		long = words

	# 	best = None
	# 	max = 0
	# 	val = 0
	# 	for w in words:

	# 	for p in itertools.permutations(long, len(short)):
	# 		value = 0
	# 		for i in range(len(short)):
	# 			if len(words)<len(received):
	# 				local = short[i]
	# 				foreign = p[i]
	# 			else:
	# 				local = p[i]
	# 				foreign = short[i]
	# 			if foreign in self.alignment and local in self.alignment[foreign]:
	# 				value += self.alignment[foreign][local]
	# 		if value >= max:
	# 			max = value
	# 			best = p

	# 	return max


	def best_match(self, protocol, possibilities, words):
		
		values = {}

		best = None
		max = 0

		for id in possibilities:
			# find best match between words and protocol[i]
			value0 = self.best_comb(protocol.text[id][0], words[0])
			value1 = self.best_comb(protocol.text[id][1], words[1])
			val = value0+value1
			# if verbose:
			# 	print val

			if val>=max:
				max = val
				best = id

		return best



	def choose_interpretation(self, protocol, restrict, interaction, received):
		# now received is a set of words
		# there is no "mappings made" anymore, because life is not bijective
		
		# now: alignment[0] are the verbs, alignment[1] are the nouns. is this necessary? NO I DONT THINK SO

		for i in [0,1]:
			for w in received[i]:
				if not w in self.alignment.keys():
					self.alignment[w] = {}

		possibilities = get_possibilities(protocol, interaction, restrict)

		if verbose:
			print "received {}".format(received)
			print possibilities
			# print "interpretation possibilities {}".format([(p, protocol.text[p]) for p in possibilities])
			# print "interpretation possibilities 0 {}".format([(p, protocol.text[p], self.best_comb(protocol.text[p][0],received[0])) for p in possibilities])
			# print "interpretation possibilities 1 {}".format([(p, protocol.text[p], self.best_comb(protocol.text[p][1],received[1])) for p in possibilities])
			print "interpretation possibilities  {}".format([(p, protocol.text[p], self.best_comb(protocol.text[p][0],received[0])+ self.best_comb(protocol.text[p][1],received[1])) for p in possibilities])

		# update values with rewards
		for i in [0,1]:
			for w in received[i]:
				for pos in possibilities:
					posT = protocol.text[pos]
					for wo in posT[i]:
						if wo in self.alignment[w]:
							self.alignment[w][wo] += 1
						else:
							self.alignment[w][wo] = 1

		# update with punishments
		for w in received[i]:
			for wo in self.alignment[w]:
				if not [1 for pos in possibilities if wo in protocol.text[pos][0] or wo in protocol.text[pos][1]]:
					self.alignment[w][wo] -= 2


		# and now choose one
		chosen = self.best_match(protocol, possibilities, received)

		return chosen


def precision_recall(alignment,  reference):
	if not alignment: 
		return 0,0
	else:
		max_alg = {k : max(alignment[k].iteritems(), key=itemgetter(1))[0] for k in alignment.keys()}
		correct = sum(1 for k in alignment.keys() if max_alg[k] == reference[k])
		return (float(correct)/float(len(alignment.keys())), float(correct)/float(len(reference.keys())))


def reverseAlg(alignment):
	return {alignment[k] : k for k in alignment.keys()}

def build_protocol(ltext, ddep):
	protocol = {}
	# if lang == 'en':
	# 	labels = list[0]
	# elif lang == 'es':
	# 	labels = list[1]
	# first the text
	vocabulary = ([],[],[],[])
	for l in ltext:
		if l[1] == 'st':
			vocabulary[0].append(l[2])
		elif l[1] == 'co':
			vocabulary[1].append(l[2])
		elif l[1] == 'nc':
			vocabulary[2].append(l[2])		
		elif l[1] == 're':
			vocabulary[3].append(l[2])

	labels = {l[2] : l[3][0] for l in ltext}

	dependencies = {}
	for l in ltext:
		if l[2] in ddep.keys():
			dependencies[l[2]] = ddep[l[2]]
		else:
			dependencies[l[2]] = []

	protocol = Protocol(vocabulary, labels, dependencies)

	return protocol

def read_dependencies():
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


def find_alignment(prot_en, prot_es):
	# alignment is list of (en, es)
	en_dep = prot_en.dependencies
	es_dep = prot_es.dependencies

	relevant_en = []
	relevant_es = []

	for lan in ((prot_en, relevant_en),(prot_es, relevant_es)):
		prot = lan[0]
		relevant = lan[1]
		for d1 in prot.dependencies.keys():
			if d1 in prot.vocabulary[0]:
				for d2 in prot.dependencies[d1]:
					if d2 in prot.vocabulary[0]:
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

	# now the alignment
	alg = [(sorted_en[i],sorted_es[i]) for i in range(len(sorted_en))]
	return alg

def best_maps(alignment):
	res = []
	for k in alignment.keys():
		if alignment[k]:
			highest = max(alignment[k].values())
			max_keys = [kk for kk in alignment[k].keys() if alignment[k][kk]==highest]

		res.append((k, max_keys, highest))

	return res

def experiment(inters, text, dependencies):
	# create agents
	a_es = Agent(0)
	a_en = Agent(1)

	# choose a protocol number
	# prot = 0

	for h in range(inters):
		
		# poss = list(range(50))
		# poss.remove(16)
		# poss.remove(43)
		# poss.remove(45)
		# poss.remove(48)
		# poss.remove(46)
		# poss.remove(47)
		# poss.remove(24)
		prot = random.choice(range(15))
		# prot = 1
		if verbose:
			print "Protocol {}".format(prot)
		# build protocols
		prot_en = build_protocol(text[prot][0], dependencies[prot][0])
		prot_es = build_protocol(text[prot][1], dependencies[prot][1])

		alg = find_alignment(prot_en, prot_es)

		k_en = random.sample(prot_en.vocabulary[0], len(prot_en.vocabulary[0])/2)
		k_es = [[p[1] for p in alg if p[0]==k][0] for k in prot_en.vocabulary[0] if not k in k_en]

		if prot_en.vocabulary[3]==[] and prot_es.vocabulary[3]==[]:
			w = random.choice([0,1])
			k_en.extend(prot_en.vocabulary[w+1])
			k_es.extend(prot_es.vocabulary[(1-w)+1])

		else:
			w = random.choice(['en', 'es'])
			if w=='en':
				k_en.extend(prot_en.vocabulary[1]+prot_en.vocabulary[2]+prot_en.vocabulary[3])
			else:
				k_es.extend(prot_es.vocabulary[1]+prot_es.vocabulary[2]+prot_es.vocabulary[3])

		patterns = [[0,1] for p in range((len(prot_en.vocabulary[0])/2)+4)]
		pattern = [e for l in patterns for e in l]

		start_interaction(a_en,a_es, prot_en, prot_es, k_en, k_es, pattern)

	print "Alignment en"
	print len(best_maps(a_en.alignment))
	print best_maps(a_en.alignment)

	# print a_en.alignment
	print ""
	print "Precision: {}".format(get_precision(best_maps(a_en.alignment), 'en'))
	print ""
	print "Alignment es"
	print best_maps(a_es.alignment)
	# print a_es.alignment
	print ""
	print "Precision: {}".format(get_precision(best_maps(a_es.alignment), 'es'))
	print ""
	print "------------------------------"
	print ""

	return


def main(argv):

	name = 'test'
	inters = 60	
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
				except:
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
	with open('data/clean_labels.json') as data_file:    
		labels = json.load(data_file)

	experiment(inters, labels, read_dependencies())

	# protocol_file = open('../input/jsonALG-alg'+str(precision)+'-'+str(recall), 'w')

	# protocol = 

	# voc_es = ['i23','i29','i33','i34','i30','i32','i21','i36','i25','i27','i38','i37','i22','i31','i20','i28','i35','i24','i26','i39']

	# voc_en = ['i16', 'i11','i13','i9','i19','i3','i8','i1','i4','i12','i0','i10','i6','i7','i14','i5','i2','i18','i17','i15']
	# prot_es = {'i23' : ['i22'], 'i29': ['i28'], 'i30': ['i29'], 'i21': ['i20','i34','i32'], 'i25': ['i24'], 'i27': ['i26'], 'i22': ['i21','i31','i36','i38'], 'i20': ['i33','i35','i37','i39'], 'i28': ['i27'], 'i24': ['i23'], 'i26': ['i25']}

	# prot_en = {'i9': ['i8'],'i3': ['i2'],'i8': ['i7'],'i1': ['i0','i14','i18'],'i4': ['i3'],'i0': ['i19', 'i12', 'i13', 'i17'],'i10': ['i9'],'i6': ['i5'],'i7': ['i6'],'i5': ['i4'],'i2': ['i1','i11', 'i15','i16']}

	# alg_es_en = {'i16': 'i31', 'i11': 'i38', 'i13': 'i37', 'i9': 'i29', 'i19': 'i33', 'i3': 'i23', 'i8': 'i28', 'i1': 'i21', 'i4': 'i24', 'i12': 'i39', 'i0': 'i20', 'i10': 'i30', 'i6': 'i26', 'i7': 'i27', 'i14': 'i32', 'i5': 'i25', 'i2': 'i22', 'i18': 'i34', 'i17': 'i35', 'i15': 'i36'}
	# alg_en_es = reverseAlg(alg_es_en)

	# k_en = random.sample(voc_en, len(voc_es)/2)
	# k_es = [alg_es_en[k] for k in voc_en if not k in k_en]

	# alg_es = {k : alg_es_en[k] for k in alg_es_en.keys() if k not in k_es}
	# alg_en = {k : alg_en_es[k] for k in alg_en_es.keys() if k not in k_en}

	# a_es = Agent(0, voc_es, prot_es, k_es)
	# a_en = Agent(1, voc_en, prot_en, k_en)


	# patterns = [[0,1] for p in range((len(voc_es)/2)+4)]
	# pattern = [e for l in patterns for e in l]

	# for h in range(inters):
	# 	print "\n Interaction {}".format(h)
	# 	start_interaction(a_en,a_es, prot_en, prot_es, k_en, k_es pattern)
	# 	# print "Result: {}".format(a1.results[-1])
	# 	print "Precision, recall es: {}".format(precision_recall(a_es.alignment, alg_es_en))
	# 	print "Precision, recall en: {}".format(precision_recall(a_en.alignment, alg_en_es))

	# print "spanish"
	# print a_es.alignment

	# print "english"
	# print a_en.alignment

global verbose 
verbose = 0

if __name__ == "__main__":
   main(sys.argv[1:])


