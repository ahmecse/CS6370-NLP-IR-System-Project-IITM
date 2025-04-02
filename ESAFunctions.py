# -*- coding: utf-8 -*-
"""
Created on Mon May  2 23:45:16 2024

@author: AHMED
"""
from mediawiki import MediaWiki
import numpy as np
def wikipediaDocs(docs_title):
        wikipedia = MediaWiki()
        wikipedia_docs = []
        for title in docs_title:
            try:
                page_list = wikipedia.search(title)
            except:
                pass
            for i in range(min(len(page_list), 5)):
                page_name = page_list[i]
                try:
                    p = wikipedia.page(page_name)
                except:
                    pass
                wikipedia_docs.append(p.summary)
        return wikipedia_docs
    
    
def get_ESA_vector(doc_list,tf_idf, DocsWiki, wordIndex, Docs, wordIndexWiki):
    esa_vec_final=np.zeros(DocsWiki.shape[0])
    for tokens in doc_list:
        tf_idf_value=0.0
        esa_vec=np.zeros(DocsWiki.shape[0])
        try:
            row_carn,col_carn=wordIndex[tokens]
            tf_idf_value=Docs[row_carn,col_carn]
        except:
            row_carn,col_carn=0,0
        try:
            row_wiki,col_wiki=wordIndexWiki[tokens]
            esa_vec=tf_idf_value*DocsWiki[:,col_wiki]
        except:
            row_wiki,col_wiki=0,0
        esa_vec_final = esa_vec_final+esa_vec
    return esa_vec_final


        