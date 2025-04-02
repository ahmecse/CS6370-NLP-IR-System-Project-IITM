from itertools import count
import re
from util import *
import math

# Add your import statements here




class Evaluation():

    def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The precision value as a number between 0 and 1
        """

        precision = -1

        #Fill in code here

        relevant_docs_count = 0

        docs_till_k = query_doc_IDs_ordered[:k]
        for i in docs_till_k:
            if i in true_doc_IDs:
                relevant_docs_count += 1
        
        precision = relevant_docs_count/k

        return precision


    def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean precision value as a number between 0 and 1
        """

        meanPrecision = -1
        sum_precision = 0

        #Fill in code here

        #preprocessing the qrels to create the relevant documents list
        #for each query

        # true_docs_all = dict()
        # for i in qrels:
        #     if int(i["query_num"]) not in true_docs_all.keys():
        #         true_docs_all[int(i["query_num"])] = []
            
        #     true_docs_all[int(i["query_num"])].append(int(i['id']))
        true_docs_all = RelDocs(qrels)



        for i in range(len(doc_IDs_ordered)):
            #this loop iterated over all the queries
            q_id = int(query_ids[i])
            predicted_docs = doc_IDs_ordered[i]

            true_docs = true_docs_all[q_id]

            precision = self.queryPrecision(predicted_docs, q_id, true_docs, k)
            sum_precision += precision
        
        meanPrecision = sum_precision/len(query_ids)

        return meanPrecision

    
    def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The recall value as a number between 0 and 1
        """

        recall = -1

        #Fill in code here
        relevant_docs_count = 0

        if len(true_doc_IDs) == 0:
            return 0
        
        docs_till_k = query_doc_IDs_ordered[:k]
        for i in docs_till_k:
            if i in true_doc_IDs:
                relevant_docs_count += 1
        
        recall = relevant_docs_count/len(true_doc_IDs)

        return recall


    def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean recall value as a number between 0 and 1
        """

        meanRecall = -1

        #Fill in code here
        sum_recall = 0

        #Fill in code here

        #preprocessing the qrels to create the relevant documents list
        #for each query

        # true_docs_all = dict()
        # for i in qrels:
        #     if int(i["query_num"]) not in true_docs_all.keys():
        #         true_docs_all[int(i["query_num"])] = []
            
        #     true_docs_all[int(i["query_num"])].append(int(i['id']))
        true_docs_all = RelDocs(qrels)



        for i in range(len(doc_IDs_ordered)):
            #this loop iterated over all the queries
            q_id = int(query_ids[i])
            predicted_docs = doc_IDs_ordered[i]

            true_docs = true_docs_all[q_id]

            recall = self.queryRecall(predicted_docs, q_id, true_docs, k)
            sum_recall += recall
        
        meanRecall = sum_recall/len(query_ids)




        return meanRecall


    def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The fscore value as a number between 0 and 1
        """

        fscore = -1

        #Fill in code here
        precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

        if precision == 0 and recall == 0:
            return 0

        fscore = (2 * precision * recall)/(precision + recall)


        return fscore


    def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value
        
        Returns
        -------
        float
            The mean fscore value as a number between 0 and 1
        """

        meanFscore = -1

        #Fill in code here
        sum_fscore = 0

        #Fill in code here

        #preprocessing the qrels to create the relevant documents list
        #for each query

        # true_docs_all = dict()
        # for i in qrels:
        #     if int(i["query_num"]) not in true_docs_all.keys():
        #         true_docs_all[int(i["query_num"])] = []
            
        #     true_docs_all[int(i["query_num"])].append(int(i['id']))
        true_docs_all = RelDocs(qrels)



        for i in range(len(doc_IDs_ordered)):
            #this loop iterated over all the queries
            q_id = int(query_ids[i])
            predicted_docs = doc_IDs_ordered[i]

            true_docs = true_docs_all[q_id]

            fscore = self.queryFscore(predicted_docs, q_id, true_docs, k)
            sum_fscore += fscore
        
        meanFscore = sum_fscore/len(query_ids)




        return meanFscore
    

    def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, qrels,k, it):
        """
        Computation of nDCG of the Information Retrieval System
        at given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg5 : int
            The k value

        Returns
        -------
        float
            The nDCG value as a number between 0 and 1
        """

        nDCG = 0

        #read qrels somehow

        #collecting relevance dictionaries for the particular query
        rel_dict = []
        for i in qrels:
            if int(i["query_num"]) == query_id:
                rel_dict.append(i)
        
        relevance_scores = []
        for i in query_doc_IDs_ordered:
            if i not in true_doc_IDs:
                relevance_scores.append(0)
            else:
                for j in rel_dict:
                    if int(j["id"]) == i:
                        relevance_scores.append(5-j["position"])
        

        ideal_relevance_scores = sorted(relevance_scores, reverse=True)
        #print(ideal_relevance_scores)
        
        docs_till_k = query_doc_IDs_ordered[:k]
        DCG = 0
        for i in range(len(docs_till_k)):
            DCG += relevance_scores[i]/(math.log2(i+2)) #added 2 here instead of 1 because list indices start from 0
        
        iDCG = 0
        for i in range(len(docs_till_k)):
            iDCG += ideal_relevance_scores[i]/(math.log2(i+2))
            
        if iDCG != 0:
            nDCG = float(DCG) / float(iDCG)


        return nDCG


    def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of nDCG of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean nDCG value as a number between 0 and 1
        """

        meanNDCG = -1

        #Fill in code here
        sum_ndcg = 0

        #Fill in code here

        #preprocessing the qrels to create the relevant documents list
        #for each query

        # true_docs_all = dict()
        # for i in qrels:
        #     if int(i["query_num"]) not in true_docs_all.keys():
        #         true_docs_all[int(i["query_num"])] = []
            
        #     true_docs_all[int(i["query_num"])].append(int(i['id']))
        true_docs_all = RelDocs(qrels)



        for i in range(len(doc_IDs_ordered)):
            #this loop iterated over all the queries
            #print("***1****")
            q_id = int(query_ids[i])
            predicted_docs = doc_IDs_ordered[i]

            true_docs = true_docs_all[q_id]

            ndcg = self.queryNDCG(predicted_docs, q_id, true_docs, qrels, k, i)
            sum_ndcg += ndcg
        
        meanNDCG = sum_ndcg/len(query_ids)


        return meanNDCG


    def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of average precision of the Information Retrieval System
        at a given value of k for a single query (the average of precision@i
        values for i such that the ith document is truly relevant)

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The average precision value as a number between 0 and 1
        """

        avgPrecision = -1
        count_relevant = 0
        sum_precision = 0
        docs_till_k = query_doc_IDs_ordered[:k]
        for i in range(len(docs_till_k)):
            if docs_till_k[i] in true_doc_IDs:
                count_relevant += 1
                sum_precision += (count_relevant)/(i+1)
        
        if count_relevant == 0 :
            return 0
            
        avgPrecision = sum_precision/count_relevant

            

        #Fill in code here

        return avgPrecision


    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of MAP of the Information Retrieval System
        at given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The MAP value as a number between 0 and 1
        """

        meanAveragePrecision = -1

        #Fill in code here
        sum_ap = 0

        #Fill in code here

        #preprocessing the qrels to create the relevant documents list
        #for each query

        # true_docs_all = dict()
        # for i in qrels:
        #     if int(i["query_num"]) not in true_docs_all.keys():
        #         true_docs_all[int(i["query_num"])] = []
            
        #     true_docs_all[int(i["query_num"])].append(int(i['id']))
        true_docs_all = RelDocs(qrels)



        for i in range(len(doc_IDs_ordered)):
            #this loop iterated over all the queries
            q_id = int(query_ids[i])
            predicted_docs = doc_IDs_ordered[i]

            true_docs = true_docs_all[q_id]

            ap = self.queryAveragePrecision(predicted_docs, q_id, true_docs, k)
            sum_ap += ap
        
        meanAveragePrecision = sum_ap/len(query_ids)




        return meanAveragePrecision

