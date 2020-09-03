from typing import Dict, Tuple

import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch


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


def load_dataset(
    dataset: str, label_column: int = 0, document_start: int = 2
) -> Tuple[np.ndarray, np.ndarray, object]:
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
        corpus_file=dataset, label_column=label_column, document_start=document_start
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


def create_dataloader(
    features: np.ndarray, labels: np.ndarray, batch_size: int = 64, num_workers: int = 0
) -> object:
    """
    Returns the data loader object for the dataset.

    Parameters
    ----------
    features: np.ndarray
        The dataset features in tensor representation.
    labels: np.ndarray
        The dataset labels in tensor representation.
    batch_size: int
        The mini-batch size to use for loading features and labels.
    num_workers: int
        The number of subprocesses to use for data.

    Returns
    -------
    data_loader: torch.utils.data.DataLoader
        The data loader object, ready for a PyTorch model use.
    """
    features = torch.from_numpy(features)
    labels = torch.from_numpy(labels)
    dataset = torch.utils.data.TensorDataset(features, labels)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )
    return data_loader
