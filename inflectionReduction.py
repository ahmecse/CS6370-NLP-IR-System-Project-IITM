from util import *
import nltk
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer


# Add your import statements here




class InflectionReduction:


	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = []

		#Fill in code here
 
		lemmatizer = nltk.WordNetLemmatizer()

		for sentence in text:
			reducedSentence = []
			tagged_words = nltk.pos_tag(sentence)
			for tagged_word in tagged_words:
				reducedSentence.append(lemmatizer.lemmatize(tagged_word[0], pos_tagger(tagged_word[1])))
			reducedText.append(reducedSentence)
		
		return reducedText



