# Advanced Information Retrieval System with NLP Enhancements

## Project Overview

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

Refer to the generated plots in the `output/` folder (e.g., `eval_plot_tfidf_vsm.png`, `eval_plot_tfidf_lsa.png`) for visual representations of the evaluation metrics for different models.

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

