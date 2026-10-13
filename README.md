# LDA-Meta-Analysis-Diagnostics

## Overview
This repository contains the data and Python source code used to perform a computational systematic review and methodological audit of diagnostic test accuracy (DTA) meta-analyses based on Artificial Intelligence (AI) in low-prevalence contexts. 

The study implements Natural Language Processing (NLP) and probabilistic topic modeling to automatically analyze scientific literature, circumventing the limitations of traditional manual reviews.

## Data
The analysis is performed on a curated corpus of 94 unique articles extracted from **PubMed** and **Web of Science**, spanning the period from 2019 to 2026. 
* Raw search strategies and queries are documented within the script.
* The pipeline includes automated deduplication using exact DOI/Title matching and similarity analysis via the `RapidFuzz` algorithm (token_sort_ratio >= 92%).
* Abstracts and institutional affiliations were programmatically retrieved using `Biopython` and the NCBI Entrez API.

**Data Availability & Copyright Note**
*Due to the proprietary licensing and Terms of Use of Clarivate's Web of Science, the raw bibliographic exports (including titles and abstracts) cannot be shared publicly in this repository. To ensure full reproducibility, we provide the exact Boolean search strings used (see methodology) and a dataset containing only the Digital Object Identifiers (DOIs) and PubMed IDs (PMIDs) of the final 94-document corpus (`lista_dois_corpus.csv`). Researchers with institutional access to WoS and PubMed can use these identifiers to seamlessly reconstruct the exact dataset used in this study.*

## Methodology and Pipeline
The workflow is entirely developed in Python and is divided into four main sequential phases:

1. **Consolidation and Filtering:** Unification of PubMed and Web of Science records, followed by rigorous deduplication and geographic normalization using `pycountry`.
2. **Text Preprocessing (NLP):** Implemented using `spaCy` (en_core_web_sm). The text undergoes cleaning, tokenization, lemmatization, Part-of-Speech filtering (keeping only nouns, adjectives, verbs, and adverbs), and removal of standard/academic stop words.
3. **Topic Modeling (LDA):** Latent Dirichlet Allocation is applied over a Bag-of-Words Document-Term Matrix (DTM) using the `Gensim` library. The optimal number of topics (k=14) was determined by maximizing semantic coherence (Cv = 0.3894).
4. **Statistical Analysis:** A methodological audit using Fisher's exact test (`SciPy`) to evaluate the statistical independence between the explicit mention of low prevalence/class imbalance and the adoption of advanced hierarchical/bivariate statistical models (e.g., HSROC).

## Dependencies
To reproduce the analysis, the following libraries are required:
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
`python -m spacy download en_core_web_sm`

## Author
* **Fabián Alejandro Celi Castillo**

## License
This project is licensed under the MIT License - see the LICENSE file for details.
