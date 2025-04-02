from util import *
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
# Add your import statements here




class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = None
		stop_words_set = set(stopwords.words('english'))
		for i in range(len(text)):
			for j in text[i]:
				if j.lower() in stop_words_set:
					text[i].remove(j)
		stopwordRemovedText = text[:]

		#Fill in code here

		return stopwordRemovedText





	