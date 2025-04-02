from heapq import merge
from util import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from numpy import dot
from numpy.linalg import norm
import numpy as np


# Add your import statements here




class InformationRetrieval():

    def __init__(self):
        self.index = None
        self.tfidfModel = None
        self.lsaModel = None
        self.finalCorpus = None
        

 
        

    def buildIndex(self, docs, docIDs, num_components = 280, vectorizer = 'tfidf'):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document
        arg2 : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """

        index = None

        #Fill in code here

        #first merging the sentences in each doc to get one big list for each doc
        merged_docs = []
        for i in range(len(docs)):
            temp = []
            for j in docs[i]:
                temp.extend(j)
            merged_docs.append(temp)

        #converting the tokens into sentences for tfidf vectorizer
        final_corpus = []
        for i in range(len(merged_docs)):
            temp = " "
            final_corpus.append(temp.join(merged_docs[i]))
        
        #final_corpus = self.finalCorpus
        
        if vectorizer == 'tfidf':
            vectorizer = TfidfVectorizer()
            vectorizer.fit(final_corpus)
            X = vectorizer.transform(final_corpus)
            self.tfidfModel = vectorizer
            doc_term_matrix = X.toarray().tolist()
            
        elif vectorizer == 'bow':
            vectorizer = CountVectorizer()
            vectorizer.fit(final_corpus)
            X = vectorizer.transform(final_corpus)
            self.tfidfModel = vectorizer
            doc_term_matrix = X.toarray().tolist()

        #write code for LSA here
        tsvd = TruncatedSVD(n_components=num_components)
        tsvd.fit(doc_term_matrix)
        tsvd_mat = tsvd.transform(doc_term_matrix)
        self.lsaModel = tsvd



        index = dict()
        for i in range(len(docIDs)):
            index[docIDs[i]] = tsvd_mat[i]


        self.index = index


    def rank(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query
        

        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """

        doc_IDs_ordered = []

        #Fill in code here
        #first merging the sentences in each doc to get one big list for each doc
        merged_queries = []
        for i in range(len(queries)):
            temp = []
            for j in queries[i]:
                temp.extend(j)
            merged_queries.append(temp)

        #converting the tokens into sentences for tfidf vectorizer
        final_queries = []
        for i in range(len(merged_queries)):
            temp = " "
            final_queries.append(temp.join(merged_queries[i]))

        vectorizer = self.tfidfModel
        X = vectorizer.transform(final_queries)
        doc_term_matrix = X.toarray().tolist()
        
        tsvd = self.lsaModel
        tsvd_mat = tsvd.transform(doc_term_matrix)
        doc_term_matrix = tsvd_mat

        for i in range(len(doc_term_matrix)):
            #here i refers to the query
            #i.e., we are iterating over each query.
            query_vector = doc_term_matrix[i]
            doc_query_sim = dict()
            doc_order = []

            #for each document, calculate cosine similarity with query
            for j in self.index.keys():
                doc_vector = self.index[j]
                cos_sim = cosine_similarity(np.reshape(query_vector, (1,-1)),np.reshape(doc_vector, (1,-1)))
                #cos_sim = dot(query_vector, doc_vector)/(norm(query_vector)*norm(doc_vector))
                doc_query_sim[j] = cos_sim

            sort_orders = sorted(doc_query_sim.items(), key=lambda x: x[1], reverse=True)
            for k in sort_orders:
                doc_order.append(k[0])

            doc_IDs_ordered.append(doc_order)
    
        return doc_IDs_ordered