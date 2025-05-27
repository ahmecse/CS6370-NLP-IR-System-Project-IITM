# NLP project: Advanced Information Retrieval System.

This repository contains the implementation of an advanced Information Retrieval (IR) system developed as part of the CS6370 course at IIT Madras. The project focuses on building and evaluating an efficient IR system leveraging various Natural Language Processing (NLP) techniques. Starting with a baseline Vector Space Model (VSM), the system explores enhancements through different preprocessing strategies, dimensionality reduction techniques like Latent Semantic Analysis (LSA), and potentially other semantic analysis methods (Explicit Semantic Analysis - ESA) and word embeddings (Word2Vec - W2V), as indicated by the codebase structure. The system is designed to retrieve relevant documents from the Cranfield collection based on user queries and includes comprehensive evaluation metrics.

## Motivation

In the age of information overload, retrieving relevant documents efficiently is a critical challenge. Traditional keyword-based search methods often struggle with semantic understanding, synonymy (different words with similar meanings), and polysemy (words with multiple meanings). This project aims to address these limitations by implementing and comparing various NLP techniques within an IR framework. The goal is to build a system that not only retrieves documents based on term matching but also understands the underlying semantic relationships, leading to more accurate and relevant search results.

## Features Implemented

This project implements several core IR and NLP components:

1.  **Modular Preprocessing Pipeline:**
    *   **Sentence Segmentation:** Options for naive splitting or using NLTK's Punkt tokenizer.
    *   **Tokenization:** Options for naive splitting or using NLTK's Penn Treebank tokenizer.
    *   **Inflection Reduction:** Lemmatization using NLTK's WordNetLemmatizer to reduce words to their base form.
    *   **Stopword Removal:** Filtering out common English stopwords using NLTK's standard list.
    *   **Custom Preprocessing (VSM-2 Enhancements):** Includes lowercasing, number removal, and punctuation/diacritics removal for improved VSM performance, as detailed in the accompanying report.

2.  **Vector Space Model (VSM):**
    *   Implementation based on Term Frequency-Inverse Document Frequency (TF-IDF) weighting using `scikit-learn`.
    *   Cosine similarity is used to rank documents against queries.
    *   Two versions (VSM-1 and VSM-2) with different preprocessing steps were evaluated (refer to the report for details), with VSM-2 showing improved performance.

3.  **Latent Semantic Analysis (LSA):**
    *   Utilizes Truncated Singular Value Decomposition (SVD) from `scikit-learn` to reduce the dimensionality of the term-document matrix (TF-IDF or Bag-of-Words).
    *   Aims to capture latent semantic relationships between terms and documents, potentially improving retrieval performance by addressing synonymy.

4.  **Explicit Semantic Analysis (ESA) & Word2Vec (W2V) Integration (Experimental):**
    *   The codebase includes dedicated files (`main_ESA.py`, `main_W2v.py`, `main_newW2V.py`, `informationRetrievalW2V.py`, `ESAFunctions.py`) suggesting implementations or experiments involving ESA and Word2Vec embeddings for information retrieval. These offer alternative semantic representation approaches.

5.  **Comprehensive Evaluation Framework:**
    *   Calculates standard IR evaluation metrics: Precision@k, Recall@k, F-score@k, Mean Average Precision (MAP)@k, and Normalized Discounted Cumulative Gain (nDCG)@k for k=1 to 10.
    *   Uses the Cranfield dataset's relevance judgments (qrels) for evaluation.
    *   Includes functionality to plot evaluation metrics against 'k'.

## Dataset

The system is designed and evaluated using the **Cranfield collection**, a standard test collection for IR research. It includes:
*   `cran_docs.json`: 1400 documents (abstracts of aerodynamics papers).
*   `cran_queries.json`: 225 queries.
*   `cran_qrels.json`: Relevance judgments indicating which documents are relevant to which queries.

## Code Structure

The repository is organized as follows:

```
├── cranfield/                  # Cranfield dataset files (docs, queries, qrels)
├── output/                     # Default directory for preprocessed files and evaluation plots
├── ESAFunctions.py             # Functions potentially related to ESA implementation
├── README.md                   # This file (will be replaced by this new version)
├── evaluation.py               # Implementation of IR evaluation metrics
├── hyperparameterTuning.py     # Script likely used for LSA hyperparameter tuning (e.g., num_components)
├── inflectionReduction.py      # Lemmatization implementation
├── informationRetrieval.py     # Core VSM (TF-IDF) implementation
├── informationRetrievalLSA.py  # Core LSA implementation (using TF-IDF or BoW)
├── informationRetrievalNewW2V.py # Potential alternative Word2Vec IR implementation
├── informationRetrievalW2V.py  # Potential Word2Vec IR implementation
├── main.py                     # Main script for running VSM (TF-IDF) model and evaluation
├── main_ESA.py                 # Main script likely for running ESA model
├── main_LSA_skl.py             # Main script for running LSA model (using scikit-learn)
├── main_W2v.py                 # Main script likely for running Word2Vec model
├── main_newW2V.py              # Main script likely for running alternative Word2Vec model
├── sentenceSegmentation.py     # Sentence segmentation implementation
├── stopwordRemoval.py          # Stopword removal implementation
├── tokenization.py             # Tokenization implementation
├── util.py                     # Utility functions (potentially shared across modules)
```

## Setup and Usage

### Prerequisites

*   Python 3.x
*   Required Python packages (install via pip):
    *   `nltk`
    *   `scikit-learn`
    *   `numpy`
    *   `matplotlib`

    ```bash
    pip install nltk scikit-learn numpy matplotlib
    ```
*   NLTK data (download required resources):
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    ```

### Running the System

The system can be run using the various `main_*.py` scripts. The primary script for the baseline VSM is `main.py`.

**1. Evaluate VSM (TF-IDF) on Cranfield Dataset:**

```bash
python main.py -dataset cranfield/ -out_folder output/
```

This command will:
*   Preprocess the Cranfield documents and queries using default settings (Punkt sentence segmenter, Penn Treebank tokenizer, Lemmatization, Stopword removal).
*   Build a TF-IDF based VSM index.
*   Rank documents for all queries.
*   Evaluate the results using Precision, Recall, F-score, MAP, and nDCG (for k=1 to 10).
*   Save preprocessed files and an evaluation plot (`eval_plot_tfidf_vsm.png`) to the `output/` directory.

**2. Evaluate LSA on Cranfield Dataset:**

Use `main_LSA_skl.py`. You might need to adjust parameters like the number of components (`num_components`) or the vectorizer (`vectorizer='tfidf'` or `vectorizer='bow'`) within the script or via command-line arguments if available (check the script's argument parser).

```bash
# Example (assuming default parameters in the script are suitable)
python main_LSA_skl.py -dataset cranfield/ -out_folder output/
```

**3. Evaluate ESA / Word2Vec Models:**

Run the corresponding `main_ESA.py`, `main_W2v.py`, or `main_newW2V.py` scripts. Examine these scripts for specific parameters or setup requirements.

```bash
# Example (syntax might vary based on script implementation)
python main_ESA.py -dataset cranfield/ -out_folder output/
python main_W2v.py -dataset cranfield/ -out_folder output/
```

**4. Handle a Custom Query (using VSM):**

```bash
python main.py -dataset cranfield/ -out_folder output/ -custom
```

The script will prompt you to enter a query, preprocess it, rank documents from the Cranfield dataset using the VSM-TFIDF model, and display the top 5 relevant document IDs.

**5. Customization:**

*   **Preprocessing:** You can change the segmenter and tokenizer via command-line arguments in `main.py`:
    ```bash
    python main.py -segmenter naive -tokenizer naive ...
    ```
*   **Output Folder:** Specify a different output directory using `-out_folder`.
*   **LSA Parameters:** Modify `num_components` and `vectorizer` directly within `informationRetrievalLSA.py` or `main_LSA_skl.py` if command-line arguments are not implemented for these.

## Evaluation Results Summary

Detailed evaluation results comparing VSM-1, VSM-2, and potentially other models (LSA, ESA, W2V) can be found in the accompanying `NLP_FINAL_REPORT.pdf`. Key findings from the report include:

*   **VSM-2 Outperforms VSM-1:** The enhanced preprocessing steps in VSM-2 (lowercasing, number/punctuation removal) led to a smaller vocabulary and consistently better performance across all metrics (Precision, Recall, F-score, MAP, nDCG) compared to VSM-1, although hypothesis testing indicated the difference might not be statistically significant at the p=0.05 level for the specific test run.
*   **LSA Performance:** The report likely discusses the impact of LSA on retrieval performance, potentially showing improvements by capturing semantic relationships but also highlighting its limitations (e.g., handling polysemy).
*   **Metric Trends:** Generally, Precision tends to decrease as 'k' increases, while Recall increases. F-score often shows a balance, and MAP/nDCG provide query-averaged performance measures.

Below are some visual representations of the evaluation metrics for different models generated by the system:

**TF-IDF VSM Evaluation:**

![TF-IDF VSM Evaluation Plot](https://private-us-east-1.manuscdn.com/sessionFile/fMBPXMgJPpGePs2ypxwGyk/sandbox/seZpd4gZ2uUOJyutU9T0Zg-images_1748340897017_na1fn_L2hvbWUvdWJ1bnR1L3Byb2plY3RfYW5hbHlzaXMvQ1M2MzcwLU5MUC1JUi1TeXN0ZW0tUHJvamVjdC1JSVRNLW1haW4vb3V0cHV0L2V2YWxfcGxvdF90ZmlkZl92c20.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvZk1CUFhNZ0pQcEdlUHMyeXB4d0d5ay9zYW5kYm94L3NlWnBkNGdaMnVVT0p5dXRVOVQwWmctaW1hZ2VzXzE3NDgzNDA4OTcwMTdfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzQnliMnBsWTNSZllXNWhiSGx6YVhNdlExTTJNemN3TFU1TVVDMUpVaTFUZVhOMFpXMHRVSEp2YW1WamRDMUpTVlJOTFcxaGFXNHZiM1YwY0hWMEwyVjJZV3hmY0d4dmRGOTBabWxrWmw5MmMyMC5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=EMvsYBJ5L346SnaWt-W2OH1A5NkGHu~vxAE62UKuZ91M6uxIMbtMAro0JCIePIsKGq6O4HMu0HhW3U4pCe~UP79K7mx7baCQYvt4hmc9mLB0cs-KKHYD~UiqirYx1Vlkd6Cw8hu~cj8s23PQQTSMxwNfE8DW~DwmdJHKHINVZmbDcNCF78vnBvi6mfIChD3Z5NdO2BmbB8wYEZR3uzoxawfZFXEi4GmpomnSj60ejLdUp9tArMxlJllRx8jpsd7koFM6J23FwxqlpPlOumcqr~EPNa3iXXizZWGWTxeRhg9SrQGJw0C61hbpC~vBQQv8egrOUtwIWUNCIY9RobJMhw__)

**TF-IDF LSA Evaluation:**

![TF-IDF LSA Evaluation Plot](https://private-us-east-1.manuscdn.com/sessionFile/fMBPXMgJPpGePs2ypxwGyk/sandbox/seZpd4gZ2uUOJyutU9T0Zg-images_1748340897017_na1fn_L2hvbWUvdWJ1bnR1L3Byb2plY3RfYW5hbHlzaXMvQ1M2MzcwLU5MUC1JUi1TeXN0ZW0tUHJvamVjdC1JSVRNLW1haW4vb3V0cHV0L2V2YWxfcGxvdF90ZmlkZl9sc2E.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvZk1CUFhNZ0pQcEdlUHMyeXB4d0d5ay9zYW5kYm94L3NlWnBkNGdaMnVVT0p5dXRVOVQwWmctaW1hZ2VzXzE3NDgzNDA4OTcwMTdfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzQnliMnBsWTNSZllXNWhiSGx6YVhNdlExTTJNemN3TFU1TVVDMUpVaTFUZVhOMFpXMHRVSEp2YW1WamRDMUpTVlJOTFcxaGFXNHZiM1YwY0hWMEwyVjJZV3hmY0d4dmRGOTBabWxrWmw5c2MyRS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=jO4EgweexouKcDHwu7KYqeJobsPVzjIZ-d0j9NTw8-4fP3mRkydeVs4PId5QQRZIDFWJIu71M323PLjZ04PeTjwEpiUue7-qZNaF1-YPv5wO5JiU6l7O9-~65NA~HQnfYSryTFJ0dJOk~Ej5DbeZJ1YyKMd2BYYlnnk-vNktan0J~3Gy3VwJuLvzF9aR~MvzN~NWihNoT4fDyGbsAj2sNtoar2pGvPLRGbSDwTPWJorqBk1IN7FFsVqIWLi3vG995FvhM3Lk3-ylPdrdo29R5aESuFDkd~C86rVdQaWlm3gWvBWT6Svz-kcHEMkwBEkJfUzw8coETfSI0ICZTIv-NQ__)

**Bag-of-Words LSA Evaluation:**

![BoW LSA Evaluation Plot](https://private-us-east-1.manuscdn.com/sessionFile/fMBPXMgJPpGePs2ypxwGyk/sandbox/seZpd4gZ2uUOJyutU9T0Zg-images_1748340897018_na1fn_L2hvbWUvdWJ1bnR1L3Byb2plY3RfYW5hbHlzaXMvQ1M2MzcwLU5MUC1JUi1TeXN0ZW0tUHJvamVjdC1JSVRNLW1haW4vb3V0cHV0L2V2YWxfcGxvdF9ib3dfbHNh.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvZk1CUFhNZ0pQcEdlUHMyeXB4d0d5ay9zYW5kYm94L3NlWnBkNGdaMnVVT0p5dXRVOVQwWmctaW1hZ2VzXzE3NDgzNDA4OTcwMThfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzQnliMnBsWTNSZllXNWhiSGx6YVhNdlExTTJNemN3TFU1TVVDMUpVaTFUZVhOMFpXMHRVSEp2YW1WamRDMUpTVlJOTFcxaGFXNHZiM1YwY0hWMEwyVjJZV3hmY0d4dmRGOWliM2RmYkhOaC5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=mgb2di3538Sr-d~SlGD9I2Tk2g1jALSzjtWsmXZSJ7C5mMRx6mTPj25WrhfQGiO5t~~qv2st9JnVimdnhsk19JTbM1I-VZ2L2gtAwaa7Bpsbsh3qmn8CBQNPI2~shuaEyO71LId09OLY8zCtkG4TnbhfdCcDAVwIIeQq3tBVTcZ1gVYujIhRQhSQvZiuhoVlbUYAImKmPAT9wywmHgWLMRDf3jfWO7y37dEFKpkns5W8Ff56mngh17Kesxm88Pi7yV~z1CW9zAscqIphmqmPa9~K1fyEcu9AB0xq~OiJLuqC0WKpJaRHxEsQ4BZ7Hhr7DLk7Lb36NMalMsgMzyDaOQ__)

**(Potential) ESA Evaluation:**

![ESA Evaluation Plot](https://private-us-east-1.manuscdn.com/sessionFile/fMBPXMgJPpGePs2ypxwGyk/sandbox/seZpd4gZ2uUOJyutU9T0Zg-images_1748340897018_na1fn_L2hvbWUvdWJ1bnR1L3Byb2plY3RfYW5hbHlzaXMvQ1M2MzcwLU5MUC1JUi1TeXN0ZW0tUHJvamVjdC1JSVRNLW1haW4vb3V0cHV0L2V2YWxfcGxvdF9FU0E.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvZk1CUFhNZ0pQcEdlUHMyeXB4d0d5ay9zYW5kYm94L3NlWnBkNGdaMnVVT0p5dXRVOVQwWmctaW1hZ2VzXzE3NDgzNDA4OTcwMThfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzQnliMnBsWTNSZllXNWhiSGx6YVhNdlExTTJNemN3TFU1TVVDMUpVaTFUZVhOMFpXMHRVSEp2YW1WamRDMUpTVlJOTFcxaGFXNHZiM1YwY0hWMEwyVjJZV3hmY0d4dmRGOUZVMEUucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzY3MjI1NjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=RmrPYGuSCebjN4ZE91FQm3EiFHcD9eQ5l8aO01xHi3wMIj2a4PEu3dwvC-W88MHXzUOXkEXcd0NRkGMzTWgl5vc0rr07r5S~6Fbh7k9XBtltvWB5yi4xB9z7MX7n3Xj3dgmB83VA3dMARPcGZOxWSH3-MVaXUskpRlExFOwb5F47D-mky8l4zCiOzSDWK0NgHqsW1PLIwEOQFkD3vEgQOvkN95VicYfsaZQaM8oc8TzQ6DGTXsjBb0LqQ0~An-y~~6uXTcqgVKFMOMdyMjX3wY7B6bAez4PFdqrHOUR~K4HR51dtEcxeYKzcT3auUylNvGPm3eGSXjgbpMrVUGR6Yw__)

**(Potential) Word2Vec LSA Evaluation:**

![W2V LSA Evaluation Plot](https://private-us-east-1.manuscdn.com/sessionFile/fMBPXMgJPpGePs2ypxwGyk/sandbox/seZpd4gZ2uUOJyutU9T0Zg-images_1748340897018_na1fn_L2hvbWUvdWJ1bnR1L3Byb2plY3RfYW5hbHlzaXMvQ1M2MzcwLU5MUC1JUi1TeXN0ZW0tUHJvamVjdC1JSVRNLW1haW4vb3V0cHV0L2V2YWxfcGxvdF93MnZfbHNh.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvZk1CUFhNZ0pQcEdlUHMyeXB4d0d5ay9zYW5kYm94L3NlWnBkNGdaMnVVT0p5dXRVOVQwWmctaW1hZ2VzXzE3NDgzNDA4OTcwMThfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzQnliMnBsWTNSZllXNWhiSGx6YVhNdlExTTJNemN3TFU1TVVDMUpVaTFUZVhOMFpXMHRVSEp2YW1WamRDMUpTVlJOTFcxaGFXNHZiM1YwY0hWMEwyVjJZV3hmY0d4dmRGOTNNblpmYkhOaC5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=Jqh7fJUy1be3g0TUxFkxw15FcwfBU~c28asV6JG3BK3PiMPxEtus3X6y~B49IWb~9BSO1HGZhi8LAgNZHxA3uhcA5wS8U3aBFWRSBLCd7y1D0em-5zscoxqLiS5i1DwQ8L-V2xtdgiHMVVBevrtctt2D-ShGJ1h4fAYfvd3bbpviO1cat2BqMdDU9A2RcFwHlbDwxso9eZzHbmjB76HbcoR9BdUB6heQPGYcjqDwSKl0qj~~Jf4hcQy7tW23ejv00~nkTOxhrSw~w8KRAq3kra9O07SufThdO3IG7nEEw49GHH5gd2koF2LSzmNon8~mP5heCG3~6LqL~HV~xodD-Q__)

**Overall Comparison (Example):**

![Final Results Comparison](output/final%20results.png)

Refer to the generated plots in the `output/` folder and the `NLP_FINAL_REPORT.pdf` for a complete analysis.

## Limitations and Future Work

Based on the report and standard IR practices, potential limitations and future directions could include:

*   **Scalability:** The current implementation might not scale efficiently to significantly larger datasets without optimizations (e.g., inverted indices, more advanced data structures).
*   **Advanced Query Expansion:** While mentioned in the report, explicit query expansion techniques (e.g., using thesauri or pseudo-relevance feedback) could be further integrated and evaluated.
*   **Advanced Semantic Models:** Exploring more sophisticated models like BERT or other transformer-based architectures for semantic matching.
*   **Clustering/Topic Modeling:** Implementing document clustering (K-Means, LDA) as discussed in the report to potentially speed up search by narrowing the search space.
*   **User Experience:** Implementing features like query auto-completion (e.g., using LSTMs as mentioned in the report) for real-time interaction.
*   **Hyperparameter Optimization:** More systematic tuning of parameters (e.g., LSA components, embedding dimensions) could yield further performance gains.

## Keywords

Information Retrieval (IR), Natural Language Processing (NLP), Vector Space Model (VSM), TF-IDF, Latent Semantic Analysis (LSA), Explicit Semantic Analysis (ESA), Word2Vec (W2V), Evaluation Metrics (Precision, Recall, F-score, MAP, nDCG), Cranfield Dataset, Text Preprocessing, Lemmatization, Tokenization.

## References

[1] G. Salton, A. Wong, and C.-S. Yang, “A vector space model for automatic indexing,” Communications of the ACM, vol. 18, no. 11, pp. 613–620, 1975.

[2] C. D. Manning, P. Raghavan, and H. Schütze, Introduction to information retrieval. Cambridge university press, 2008.

[3] S. Deerwester, S. T. Dumais, G. W. Furnas, T. K. Landauer, and R. Harshman, “Indexing by latent semantic analysis,” Journal of the American society for information science, vol. 41, no. 6, pp. 391–407, 1990.

[4] T. K. Landauer, P. W. Foltz, and D. Laham, “An introduction to latent semantic analysis,” Discourse processes, vol. 25, no. 2-3, pp. 259–284, 1998.

[5] T. Hofmann, “Probabilistic latent semantic indexing,” in Proceedings of the 22nd annual international ACM SIGIR conference on Research and development in information retrieval, pp. 50–57, 1999.

[6] C. Carpineto and G. Romano, “A survey of automatic query expansion in information retrieval,” Acm Computing Surveys (CSUR), vol. 44, no. 1, pp. 1–50, 2012.

[7] H. K. Azad and A. Deepak, “Query expansion techniques for information retrieval: a survey,” Information Processing & Management, vol. 56, no. 5, pp. 1698–1735, 2019.

[8] J. A. Hartigan and M. A. Wong, “Algorithm as 136: A k-means clustering algorithm,” Journal of the royal statistical society. series c (applied statistics), vol. 28, no. 1, pp. 100–108, 1979.

[9] I. Sutskever, O. Vinyals, and Q. V. Le, “Sequence to sequence learning with neural networks,” Advances in neural information processing systems, vol. 27, 2014.

[10] D. M. Blei, A. Y. Ng, and M. I. Jordan, “Latent dirichlet allocation,” Journal of machine Learning research, vol. 3, no. Jan, pp. 993–1022, 2003.



