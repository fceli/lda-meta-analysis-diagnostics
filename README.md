# lda-meta-analysis-diagnostics
Code and datasets for applying Natural Language Processing (NLP) and Latent Dirichlet Allocation (LDA) to optimize meta-analysis of diagnostic tests.
# LDA-Meta-Analysis-Diagnostics

## Overview
This repository contains the data and Python source code used to perform a computational systematic review and methodological audit of diagnostic test accuracy (DTA) meta-analyses based on Artificial Intelligence (AI) in low-prevalence contexts[cite: 2, 3]. 

The study implements Natural Language Processing (NLP) and probabilistic topic modeling to automatically analyze scientific literature, circumventing the limitations of traditional manual reviews[cite: 3, 14].

## Data
The analysis is performed on a curated corpus of 94 unique articles extracted from **PubMed** and **Web of Science**, spanning the period from 2019 to 2026[cite: 3, 31]. 
* Raw search strategies and queries are documented within the script[cite: 31].
* The pipeline includes automated deduplication using exact DOI/Title matching and similarity analysis via the `RapidFuzz` algorithm (token_sort_ratio $\ge$ 92%)[cite: 31, 34].
* Abstracts and institutional affiliations were programmatically retrieved using `Biopython` and the NCBI Entrez API[cite: 31, 34].

## Methodology and Pipeline
The workflow is entirely developed in Python and is divided into four main sequential phases[cite: 30, 31]:

1. **Consolidation and Filtering:** Unification of PubMed and Web of Science records, followed by rigorous deduplication and geographic normalization using `pycountry`[cite: 34].
2. **Text Preprocessing (NLP):** Implemented using `spaCy` (en_core_web_sm). The text undergoes cleaning, tokenization, lemmatization, Part-of-Speech filtering (keeping only nouns, adjectives, verbs, and adverbs), and removal of standard/academic stop words[cite: 33].
3. **Topic Modeling (LDA):** Latent Dirichlet Allocation is applied over a Bag-of-Words Document-Term Matrix (DTM) using the `Gensim` library[cite: 33, 35]. The optimal number of topics ($k=14$) was determined by maximizing semantic coherence ($C_v = 0.3894$)[cite: 3, 36].
4. **Statistical Analysis:** A methodological audit using Fisher's exact test (`SciPy`) to evaluate the statistical independence between the explicit mention of low prevalence/class imbalance and the adoption of advanced hierarchical/bivariate statistical models (e.g., HSROC)[cite: 30, 37].

## Dependencies
To reproduce the analysis, the following libraries are required[cite: 30, 53, 54, 55, 59, 62]:
* `pandas`
* `numpy`
* `spacy`
* `gensim`
* `scipy`
* `matplotlib`
* `biopython`
* `rapidfuzz`
* `geopandas`
* `pycountry`
* `seaborn`
* `scikit-learn`

*Note: The spaCy English language model must be downloaded before execution:*
`python -m spacy download en_core_web_sm`[cite: 53]

## Author
* **Fabián Alejandro Celi Castillo**[cite: 2]

## License
This project is licensed under the MIT License - see the LICENSE file for details.
