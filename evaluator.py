# import urllib2
from HTMLParser import HTMLParser

# response = urllib2.urlopen('http://python.org/')
# html = response.read()

from lxml import html
import requests
import csv
import nltk
import pandas
import urllib2
from json import loads
import math
import numpy as np
from nltk.stem import SnowballStemmer
import re
from bs4 import BeautifulSoup




def get_precision(alignment, language):

	good = []
	bad = []

	if language == 'es':
		lang_ad = 'es/en'
		lfrom = 'eng'
		lto = 'es'
		stemmer = SnowballStemmer("spanish")
	
	else:
		lang_ad = 'es'
		lfrom = 'es'
		lto = 'eng'
		stemmer = SnowballStemmer("english")
	for pair in alignment:
		try:
			# url = 'http://www.wordreference.com/{}/translation.asp?tranword={}'.format(lang_ad,pair[0])
			url = 'https://glosbe.com/gapi/translate?from={}&dest={}&format=json&phrase={}&pretty=true'.format(lfrom,lto,pair[0])
			# print url
			# page_def = requests.get('http://www.wordreference.com/{}/translation.asp?tranword={}'.format(lang_ad,pair[0]))
			page_json = requests.get(url)
			# page_def = html.fromstring(page_def)
			json_data = page_json.json()
			# tree_trans = html.fromstring(page_trans.content)
		except requests.exceptions.RequestException as e:    # This is the correct syntax
			print "Error {}".format(pair)
			sys.exit(1)
				
		# soup = BeautifulSoup(page_def.content, "lxml")
		# spans = soup.find_all('span', {'class' : 'ToWrd'})
		# if not spans:
		# 	print pair[0]
		# 	bad.append(pair)

		# found = False
		# for sp in spans:
		# 	if not found:
		# 		trans_wr = sp.get_text()
		# 		for w in pair[1]:
		# 			if not found:
		# 				trans = stemmer.stem(w)
		# 				if startswith(trans, trans_wr):
		# 					good.append(pair)
		# 					found = True
		# if not found:
		# 	bad.append(pair)
 
		if not 'tuc' in json_data.keys():
			print "Not found {}".format(pair[0])
			bad.append(pair)

		else:
			found = False
			for sp in json_data['tuc']:
				if not found and 'phrase' in sp:
					trans_wr = stemmer.stem(sp['phrase']['text'])
					for w in pair[1]:
						if not found:
							trans = stemmer.stem(w)
							if trans.startswith(trans_wr):
								good.append(pair)
								found = True
			if not found:
				bad.append(pair)

	# print "Good {}".format(good)
	# print ""
	# print "Bad {}".format(bad)
	# print ""

	return len(good)/float(len(alignment))


from PyDictionary import PyDictionary

dictionary=PyDictionary()
print (dictionary.translate("apron",'es'))

