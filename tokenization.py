from util import *
import re
from nltk.tokenize import TreebankWordTokenizer
# Add your import statements here




class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		#Fill in code here
		for i in text:
			temp = text.split()
			tokenizedText.append(temp[:len(temp)-1])

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		#Fill in code here
		for i in text:
			temp = TreebankWordTokenizer().tokenize(i)
			tokenizedText.append(temp[:len(temp)-1])
        
		#tokenizedText = TreebankWordTokenizer().tokenize(text)
		#tokenizedText = tokenizedText[:len(tokenizedText)-1]
		

		return tokenizedText
