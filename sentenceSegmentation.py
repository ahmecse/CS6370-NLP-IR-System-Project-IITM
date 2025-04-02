from util import *
import re
from nltk.tokenize import sent_tokenize
import nltk.data
nltk.download('punkt')

# Add your import statements here




class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None

		regex = re.compile('[.!?]')
		segmentedText = regex.split(text)
		segmentedText = segmentedText[:len(segmentedText)-1]


		return segmentedText





	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None

		sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
		segmentedText = sentence_detector.tokenize(text.strip())

		#Alternatively below code could also be used. 
		#sent_tokenize also makes use of nltk's punkt to segment the setence

		# segmentedText = nltk.sent_tokenize(text)



		#Fill in code here
		
		return segmentedText