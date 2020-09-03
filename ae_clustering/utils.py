from typing import Dict, Tuple

import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def read_data(corpus_file: str, label_column: int, document_start: int) -> Dict:
    """
    Returns a <key, value> pair of the loaded dataset
    where the key is the text data and the value is the data label.

    Parameters
    ----------
    corpus_file: str
        The filename of the dataset to load.
    label_column: int
        The column number of the dataset label (zero-indexed).
    document_start: int
        The number of columns in the dataset.

    Returns
    -------
    dataset: Dict
        The <key, value> pair representing the text data and their labels.
    """
    dataset = {}
    with open(corpus_file, "r", encoding="utf-8") as text_data:
        for line in text_data:
            columns = line.strip().split(maxsplit=document_start)
            text = columns[-1]
            label = int(columns[label_column].strip("__label__"))
            dataset[text] = label
    return dataset


def load_dataset(dataset: str, label_column: int = 0, doc_start: int = 2) -> Tuple:
    """
    Loads the dataset from file.

    Parameters
    ----------
    dataset: str
        The filename of the dataset to load.
    label_column: int
        The column number of the dataset label (zero-indexed).
    document_start: int
        The number of columns in the dataset.

    Returns
    -------
    vectors: np.ndarray
        The TF-IDF vector representation of the text data.
    labels: np.ndarray
        The labels of the text data.
    vectorizer: sklearn.feature_extraction.text.TfidfVectorizer
        The vectorizer object used for computing the
        vector representation of the text data.
    """
    dataset = read_data(
        corpus_file=dataset, label_column=label_column, doc_start=doc_start
    )
    texts = dataset.keys()
    labels = dataset.values()
    texts = list(map(lambda text: text.replace("[^a-zA-Z#]", ""), texts))
    texts = list(
        map(
            lambda text: " ".join([word for word in text.split() if len(word) > 3]),
            texts,
        )
    )
    texts = list(map(lambda text: text.lower(), texts))
    texts = list(map(lambda text: text.split(), texts))
    en_stopwords = nltk.corpus.stopwords.words("english")
    texts = list(
        map(lambda text: [word for word in text if word not in en_stopwords], texts)
    )
    texts = list(map(lambda text: " ".join(text), texts))
    vectorizer = TfidfVectorizer(
        max_features=2000, sublinear_tf=True, max_df=0.5, smooth_idf=True
    )
    vectors = vectorizer.fit_transform(texts)
    vectors = vectors.toarray()
    vectors = vectors.astype(np.float32)
    labels = np.array(list(labels), dtype=np.float32)
    return (vectors, labels, vectorizer)
