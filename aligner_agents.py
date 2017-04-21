
#!/usr/bin/python

import sys, getopt
from multiprocessing import Process, Pipe, Queue
import random
#from scipy.stats import rv_discrete

from operator import itemgetter
import numpy as np
import itertools

# verbose = 0
# if remote:
# 	verbose = 0

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def start_interaction(agent1, agent2, prot1, prot2, pattern):
	""" Starts interaction between two agents"""
	first_conn, second_conn = Pipe()
	queue = Queue()
	result_1 = []
	result_2 = []
	a1 = Interlocutor(agent1, first_conn, queue, prot1, pattern)
  	a2 = Interlocutor(agent2, second_conn, queue, prot2, pattern)

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


class Interlocutor(Process):
	""" An interlocutor that relates a process with an agent"""
	
	def __init__(self, agent, connection, queue, protocol, pattern):
		super(Interlocutor, self).__init__()
		self.agent = agent
		self.connection = connection
		self.queue = queue
		self.protocol = protocol
		self.pattern = pattern

	def run(self):
		result = self.agent.interact(self.connection, self.protocol, self.pattern)
		if verbose:
			print "outcome {}".format(result)
		# this should also be an agent's method (something like "remember")
		self.queue.put([self.agent.id, self.agent.alignment])
		return

# a protocol is dict : ids -> ((verbs, nouns), [ids]) which mean text, dependencies
# and interaction is [ids]

def is_possible(id, protocol, interaction):
	if not id in protocol[0].keys():
		return False
	if not id in protocol[1].keys():
		# there are no dependencies
		return True
	else:		
		dependencies = protocol[id][1]
	return not [d for d in dependencies if not d in interaction]

def get_possibilities(protocol, interaction):
	return [l for l in protocol.keys() if is_possible(l, protocol, interaction)]


class Agent():
	""" A basic agent"""
	def __init__(self, id, vocabulary, known):
		self.id = id
		self.vocabulary = vocabulary
		# self.protocol = protocol
		self.alignment = {}
		self.known = known
		self.unknown = [k for k in self.vocabulary if not k in self.known]
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


	def interact(self, connection, protocol, pattern):
		"""Start an interaction with an agent"""
		interaction = []
		bound = len(pattern)
		mappings_made = {}
		for t in pattern: 
			if t==self.id:
				utterance = self.choose_utterance(protocol, interaction)
				if not utterance:
					if interaction and interaction[-1] == 'none':
						connection.send('failed')
						if verbose:
							print "failed by sender"
						return 0
					else:
						connection.send('none')
						if verbose:
							print "sent  none"
						return 0
				connection.send(utterance)
				if verbose:
					print "Agent {} says {}".format(self.id, utterance)
				interaction.append(utterance)
				conf = connection.recv()
				if conf == 'failed':
					return 0
			else:
				received = connection.recv()
				if received == 'failed':
					return 0		
				
 				interpretation = self.choose_interpretation(protocol, interaction, received, mappings_made)	
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

	def choose_utterance(self, protocol, interaction):
		poss = get_possibilities(protocol, interaction)
		if poss: 
			# return the text
			return protocol[random.choice(poss)][0]
		return None
	
	def best_comb(words, received):
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
			if value > max:
				max = value
				best = p

		return max


	def best_match(self, protocol, possibilities, words):
		
		values = {}

		best = None
		max = 0

		for id in possibilities:
			# find best match between words and protocol[i]
			value0 = self.best_comb(protocol[id][0][0], words[0])
			value1 = self.best_comb(protocol[id][0][1], words[1])
			val = value0+value1

			if val>max:
				max = val
				best = id

		return best



	def choose_interpretation(self, protocol, interaction, received):
		# now received is a set of words
		# there is no "mappings made" anymore, because life is not bijective
		
		# now: alignment[0] are the verbs, alignment[1] are the nouns. is this necessary? NO I DONT THINK SO

		for i in [0,1]:
			for w in received[i]:
				if not w in self.alignment.keys():
					self.alignment = []

		possibilities = get_possibilities(protocol, interaction)

		# update values
		for i in [0,1]:
			for w in received[i]:
				for pos in possibilities:
					for wo in pos[i]:
						if wo in self.alignment[w]:
							self.alignment[w][wo] += 1
						else:
							self.alignment[w][wo] = 1

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
	protocol = {l[2] : (l[3][0], ddep[l[2]]) for l in ltext}

	return protocol

def read_dependencies():
	with open('all_dependencies.txt', encoding='utf-8') as f:
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
            print iprot        
            iprot += 1
            labels_store.append([{},{}])
        elif l.startswith('#PAIR'):
        	jpair = 1 - jpair
        elif not l.startswith("#"):
        	lab, dep = l.split()
        	if lab in dependencies[iprot][jpair]:
        		dependencies[iprot][jpair].append(dep)
        	else:
        		dependencies[iprot][jpair] = [dep]
	return dependencies


def experiment(inters, text, dependencies):

	# create agents
	a_es = Agent(0)
	a_en = Agent(1)

	# solve the partitions, maybe inside the protocols?


	# choose a protocol number
	prot = random.choose(range(650))
	# build protocols
	prot_en = build_protocol(text[prot][0], dependencies[prot][0])
	prot_en = build_protocol(text[prot][1], dependencies[prot][1])

	



def main(argv):

	name = 'test'
	size = 50
	inters = 100
	agentType = 3
	ver = 0
	prot = 'europe'

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

	# protocol_file = open('../input/jsonALG-alg'+str(precision)+'-'+str(recall), 'w')

	# protocol = 

	voc_es = ['i23','i29','i33','i34','i30','i32','i21','i36','i25','i27','i38','i37','i22','i31','i20','i28','i35','i24','i26','i39']

	voc_en = ['i16', 'i11','i13','i9','i19','i3','i8','i1','i4','i12','i0','i10','i6','i7','i14','i5','i2','i18','i17','i15']
	prot_es = {'i23' : ['i22'], 'i29': ['i28'], 'i30': ['i29'], 'i21': ['i20','i34','i32'], 'i25': ['i24'], 'i27': ['i26'], 'i22': ['i21','i31','i36','i38'], 'i20': ['i33','i35','i37','i39'], 'i28': ['i27'], 'i24': ['i23'], 'i26': ['i25']}

	prot_en = {'i9': ['i8'],'i3': ['i2'],'i8': ['i7'],'i1': ['i0','i14','i18'],'i4': ['i3'],'i0': ['i19', 'i12', 'i13', 'i17'],'i10': ['i9'],'i6': ['i5'],'i7': ['i6'],'i5': ['i4'],'i2': ['i1','i11', 'i15','i16']}

	alg_es_en = {'i16': 'i31', 'i11': 'i38', 'i13': 'i37', 'i9': 'i29', 'i19': 'i33', 'i3': 'i23', 'i8': 'i28', 'i1': 'i21', 'i4': 'i24', 'i12': 'i39', 'i0': 'i20', 'i10': 'i30', 'i6': 'i26', 'i7': 'i27', 'i14': 'i32', 'i5': 'i25', 'i2': 'i22', 'i18': 'i34', 'i17': 'i35', 'i15': 'i36'}
	alg_en_es = reverseAlg(alg_es_en)

	k_en = random.sample(voc_en, len(voc_es)/2)
	k_es = [alg_es_en[k] for k in voc_en if not k in k_en]

	alg_es = {k : alg_es_en[k] for k in alg_es_en.keys() if k not in k_es}
	alg_en = {k : alg_en_es[k] for k in alg_en_es.keys() if k not in k_en}

	a_es = Agent(0, voc_es, prot_es, k_es)
	a_en = Agent(1, voc_en, prot_en, k_en)


	patterns = [[0,1] for p in range((len(voc_es)/2)+4)]
	pattern = [e for l in patterns for e in l]

	for h in range(inters):
		print "\n Interaction {}".format(h)
		start_interaction(a_es,a_en, pattern)
		# print "Result: {}".format(a1.results[-1])
		print "Precision, recall es: {}".format(precision_recall(a_es.alignment, alg_es_en))
		print "Precision, recall en: {}".format(precision_recall(a_en.alignment, alg_en_es))

	print "spanish"
	print a_es.alignment

	print "english"
	print a_en.alignment

global verbose 
verbose = 0

if __name__ == "__main__":
   main(sys.argv[1:])


