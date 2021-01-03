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
import string
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import nltk
import numpy as np
from scipy.optimize import linear_sum_assignment
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from tsnecuda import TSNE

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
    texts = list(
        map(
            lambda text: text.translate(str.maketrans("", "", string.punctuation)),
            texts,
        )
    )
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
        ngram_range=(1, 3), max_features=2000, max_df=0.5, smooth_idf=True
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


def compute_latent_code(model: torch.nn.Module, features: np.ndarray) -> np.ndarray:
    """
    Computes the latent code representation for the features
    using a (presumably) trained autoencoder network.

    Parameters
    ----------
    model: torch.nn.Module
        The autoencoder network.
    features: np.ndarray
        The features to represent in latent space.

    Returns
    -------
    latent_code: np.ndarray
        The latent code representation for the features.
    """
    features = torch.from_numpy(features)
    activations = {}
    for index, layer in enumerate(model.encoder_layers):
        if index == 0:
            activations[index] = layer(features)
        else:
            activations[index] = layer(activations[index - 1])
    latent_code = activations[len(activations) - 1]
    latent_code = latent_code.detach().numpy()
    return latent_code


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
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.numpy()
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


def display_results(results: str) -> None:
    """
    Prints the clustering results with the column (metric) names.

    Parameter
    ---------
    results: str
        The formatted string of clustering results.
    """
    print(120 * "_")
    print("model\t\ttime\tdb-index\tsilhouette\tch-score\t\tnmi\t\tari\t\tacc")
    print(results)
    print(120 * "_")
    return None


def display_latent_code(
    latent_code: np.ndarray, labels: np.ndarray, title: str, seed: int
) -> None:
    """
    Plots the computed latent code representation for the features.

    Parameters
    ----------
    latent_code: np.ndarray
        The latent code representation for features.
    labels: np.ndarray
        The labels for the dataset features.
    title: str
        The plot title to use.
    seed: int
        The pseudorandom seed to use for reproducible t-SNE visualization.
    """
    tsne_encoder = TSNE(random_seed=seed, perplexity=50, learning_rate=10, n_iter=5000)
    latent_code = tsne_encoder.fit_transform(latent_code)
    sns.set_style("darkgrid")
    plt.scatter(latent_code[:, 0], latent_code[:, 1], c=labels, marker="o")
    plt.title(title)
    plt.grid()
    plt.savefig(fname=f"data/{title}.png", dpi=150)
    plt.show()


def export_vectorizer(vectorizer: object, filename: str = "data/vectorizer.pk") -> None:
    """
    Exports the vectorizer object to file.

    Parameters
    ----------
    vectorizer: sklearn.feature_extraction.text.TfidfVectorizer
        The vectorizer to export.
    filename: str
        The filename to use for the vectorizer file.
    """
    with open(filename, "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)


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


def load_clustering_model(filename: str = "models/kmeans.pk") -> object:
    """
    Loads the fitted k-Means clustering model from file.

    Parameter
    ---------
    filename: str
        The path to the k-means clustering (pickle) model file.

    Returns
    -------
    model: sklearn.cluster._kmeans.KMeans
        The exported k-Means clustering model.
    """
    with open(filename, "rb") as kmeans_pickle:
        model = pickle.load(kmeans_pickle)
    return model


def vectorize_text(text: str, vectorizer: object) -> np.ndarray:
    """
    Returns the TF-IDF vector representation of a text.

    Parameters
    ----------
    text: str
        The input text to vectorize.
    vectorizer: sklearn.feature_extraction.text.TfidfVectorizer
        The vectorizer to use for the input text.

    Returns
    -------
    vector: np.ndarray
        The vector representation of the input text.
    """
    vector = vectorizer.transform([text])
    vector = vector.toarray()
    vector = vector.astype(np.float32)
    return vector


def cluster_text(
    vector: np.ndarray, autoencoder_model: object, kmeans_model: object
) -> np.ndarray:
    """
    Returns the cluster number for the input text.

    Parameters
    ----------
    vector: np.ndarray
        The text vector to cluster.
    autoencoder_model: torch.nn.Module
        The trained autoencoder to use for computing
        the latent code representation
        of the input text.
    kmeans_model: sklearn.cluster._kmeans.KMeans
        The fitted k-Means clustering model.

    Returns
    -------
    cluster_index: np.ndarray
        The cluster index of the input text.
    """
    latent_code = autoencoder_model.compute_latent_code(features=vector)
    cluster_index = kmeans_model.predict(latent_code)
    return cluster_index
