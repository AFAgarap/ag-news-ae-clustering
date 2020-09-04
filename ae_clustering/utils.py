# AG News Clustering with Autoencoder
# Copyright (C) 2020  Abien Fred Agarap
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Utility functions module"""
import pickle
from typing import Dict, Tuple

import nltk
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

__author__ = "Abien Fred Agarap"


def set_global_seed(seed: int) -> None:
    """
    Sets the pseudorandom seed for reproducibility.

    Parameter
    ---------
    seed: int
        The pseudorandom seed to use.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return None


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


def clustering_accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
    """
    Returns the clustering accuracy.

    The clustering accuracy metric is based on
    Guo et al., 2018
    [http://proceedings.mlr.press/v95/guo18b/guo18b.pdf]

    Parameters
    ----------
    target: np.ndarray
        The set of true labels.
    prediction: np.ndarray
        The set of predicted (pseudo) labels / clusters

    Returns
    -------
    float
        The clustering accuracy.
    """
    target = target.astype(np.int64)
    assert target.size == prediction.size
    D = max(prediction.max(), target.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for index in range(prediction.size):
        w[prediction[index], target[index]] += 1

    indices = linear_sum_assignment(w.max() - w)
    indices = np.asarray(indices)
    indices = np.transpose(indices)

    return sum([w[i, j] for i, j in indices]) * 1.0 / prediction.size


def load_vectorizer(filename: str = "data/vectorizer.pk") -> object:
    """
    Loads the exported vectorizer from file.

    Parameter
    ---------
    filename: str
        The path to the vectorizer (pickle) file.

    Returns
    -------
    vectorizer: sklearn.feature_extraction.text.TfidfVectorizer
        The exported vectorizer.
    """
    with open(filename, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return vectorizer
