# Add your import statements here

import nltk
from nltk.corpus import wordnet
from itertools import chain
from collections import Counter
import numpy as np
import math



def pos_tagger(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

def RelDocs(qrels):
    true_docs_all = dict()

    for i in qrels:
        if int(i["query_num"]) not in true_docs_all.keys():
            true_docs_all[int(i["query_num"])] = []
            
        true_docs_all[int(i["query_num"])].append(int(i['id']))
    return true_docs_all


'''def install():
    pip install pymediawiki
    pip install pymediawiki==0.6.7'''

def chain_lists(processedDocs):
    all_docs=[]
    for i in range(len(processedDocs)):
        all_docs.append(list(chain.from_iterable(processedDocs[i])))
    return all_docs
# Add any utility functions here

def docFreqCalc(all_docs):
    # create a dictionary of key-value pairs where tokens are keys and their occurence in the corpus the value
    DF = {}
    no_of_docs=len(all_docs)
    for i in range(no_of_docs):
        tokens = all_docs[i]
        for w in tokens:
            try:
                # add token as key and doc number as value is chained
                DF[w].add(i)
            except:
                # to handle when a new token is encountered
                DF[w] = {i}

    for i in DF:
        # convert to number of occurences of the token from list of documents where token occurs
        DF[i] = len(DF[i])
    #print(DF)
    return DF


def make_dictionary(DF):
    # count number of unique words in the corpus
    vocab_size = len(DF)
    #print(vocab_size)
    # create vocabulary list of all unique words
    vocab = [term for term in DF]
    #print(vocab)
    return vocab,vocab_size


def create_tf_idf(all_docs,DF,vocab):
    doc = 0

    # creating dictionary to store tf-idf values for each term in the vocabulary
    tf_idf = {}
    no_of_docs=len(all_docs)
    for i in range(no_of_docs):
        
        tokens = all_docs[i]
        
        # counter object to efficiently count number of occurence of a term in a particular document
        counter = Counter(tokens)
        words_count = len(tokens)
        
        for token in np.unique(tokens):
            
            # counting occurence of term in object using counter object
            tf = counter[token]/words_count
            # retrieving df values from DF dictionary
            df = DF[token] if token in vocab else 0
            
            # adding 1 to numerator & denominator to avoid divide by 0 error
            idf = np.log((no_of_docs+1)/(df+1))
            
            tf_idf[doc, token] = tf*idf

        doc += 1
    return tf_idf


def createDocMatrix(no_of_docs,vocab_size,tf_idf,vocab):
    # initializing empty vector of vocabulary size
    D = np.zeros((no_of_docs, vocab_size))
    word_index={}
    # creating vector of tf-idf values
    for i in tf_idf:
        ind = vocab.index(i[1])
        D[i[0]][ind] = tf_idf[i]
        word_index[i[1]] =i[0],ind
    #print(len(D),D.shape)
    #print(D)
    return D,word_index

def genQueryVector(tokens,vocab,DF,no_of_docs):
    """To create a vector (with repsect to the vocabulary) of the tokens passed as input
    
    Arguments:
        tokens {list} -- list of tokens to be converted
    
    Returns:
        numpy.ndarray -- vector of tokens
    """
    Q = np.zeros((len(vocab)))
    
    counter = Counter(tokens)
    words_count = len(tokens)

    query_weights = {}
    
    for token in np.unique(tokens):
        
        tf = counter[token]/words_count
        df = DF[token] if token in vocab else 0
        idf = math.log10((no_of_docs+1)/(df+1))

        try:
            ind = vocab.index(token)
            Q[ind] = tf*idf
        except:
            pass
    #print(Q.shape)
    return Q


def normalize_cols(x: np.ndarray):
    """
    function that normalizes each row of the matrix x to have unit length.

    Args:
     ``x``: A numpy matrix of shape (n, m)

    Returns:
     ``x``: The normalized (by row) numpy matrix.
    """
    return x/(np.linalg.norm(x, ord=2, axis=0, keepdims=True)+0.00000000000000000001)


def return_cosine_similarity(D_rep,Q_rep):
    Doc_tf_id=D_rep.T
    #print(Doc_tf_id.shape)
    Q_rep = Q_rep.T
    Q_rep=normalize_cols(Q_rep)
    Doc_tf_id=normalize_cols(Doc_tf_id)
    #print(Doc_tf_id.shape)
    #print(Q_rep.shape)
    cosine_smiliratity_values = (np.dot( np.transpose(Doc_tf_id),Q_rep))
    #print(cosine_smiliratity_values.shape)
    return  cosine_smiliratity_values

def return_orderded_docs(cosine_smiliratity_values):
    b= cosine_smiliratity_values
    #print(b,"\n")
    out = np.array(b.argsort(axis=0))[::-1]
    #print(out,"\n")
    #b.sort(axis=0)
    #print(b,"\n")
    b.shape
    doc_IDs_ordered = (out.T).tolist()
    print(len(doc_IDs_ordered))
    for i in range(len(doc_IDs_ordered)):
        res=doc_IDs_ordered[i]
        doc_IDs_ordered[i]=[x + 1 for x in res]
    print(doc_IDs_ordered[0])
    return doc_IDs_ordered