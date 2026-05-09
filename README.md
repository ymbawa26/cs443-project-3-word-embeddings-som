# Word Embeddings and Self-Organizing Maps

[![Open demo](https://img.shields.io/badge/Open%20demo-GitHub%20Pages-111827)](https://ymbawa26.github.io/cs443-project-3-word-embeddings-som/)
[![Open in Colab](https://img.shields.io/badge/Open%20notebook-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/ymbawa26/cs443-project-3-word-embeddings-som/blob/main/word_embeddings.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Skip--gram-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

Recruiter-friendly readout of a CS 443 Bio-inspired Machine Learning project. The core idea: learn word vectors from movie-review text, then use a self-organizing map to make the learned structure visible.

## What This Shows

- Implemented a Skip-gram neural network for word embeddings.
- Built text preprocessing utilities for IMDb-style review text.
- Added cosine-similarity search over learned embeddings.
- Implemented a self-organizing map with BMU lookup, Gaussian neighborhoods, weight updates, and U-matrix visualization.
- Completed notebooks for both embedding training and SOM-based word-cloud analysis.

## Quick Results

| Check | Result |
| --- | --- |
| Embedding vectors loaded | 141 word vectors |
| Similarity test query | `Waterville` -> `Bangor`, `Camden`, `Portland` |
| IMDb query sample | `movie` -> `loved`, `cast`, `humor`, `emotion`, `script` among nearby terms |
| SOM visualization | Iris U-matrix and IMDb embedding word cloud |

## What To Open First

- [Project demo page](https://ymbawa26.github.io/cs443-project-3-word-embeddings-som/) gives the fastest overview.
- [Word embeddings notebook](https://github.com/ymbawa26/cs443-project-3-word-embeddings-som/blob/main/word_embeddings.ipynb) shows preprocessing and Skip-gram training.
- [Word cloud notebook](https://github.com/ymbawa26/cs443-project-3-word-embeddings-som/blob/main/word_cloud.ipynb) shows SOM implementation and visualization.
- [Open the embeddings notebook in Colab](https://colab.research.google.com/github/ymbawa26/cs443-project-3-word-embeddings-som/blob/main/word_embeddings.ipynb) to run it online.

## Files

- `word_embeddings.ipynb` walks through text preprocessing, Skip-gram training, and embedding analysis.
- `word_cloud.ipynb` uses trained embeddings and SOM-style visualization.
- `skipgram.py`, `skipgram_layers.py`, `som.py`, and `text_dataset_word.py` contain the main implementations.
- `data/imdb_train.csv` is a small included dataset for reproducible notebook runs.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Validation

The notebooks and helper files were previously repaired and validated locally so the assignment cells run with the expected outputs.
