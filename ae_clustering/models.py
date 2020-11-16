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
"""Implementation of autoencoder and clustering models"""
import os
import pickle
import time
from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import silhouette_score
import torch

from ae_clustering.utils import clustering_accuracy


__author__ = "Abien Fred Agarap"


class Autoencoder(torch.nn.Module):
    def __init__(
        self, input_shape: int = 784, code_dim: int = 128, learning_rate: float = 1e-4
    ):
        """
        Constructs the autoencoder model.

        Parameters
        ----------
        input_shape: int
            The dimensionality of the input features.
        code_dim: int
            The dimensionality of the latent code.
        learning_rate: float
            The learning rate to use for optimization.
        """
        super().__init__()
        self.model_device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.encoder_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_features=input_shape, out_features=500),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=500, out_features=500),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=500, out_features=2000),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2000, out_features=code_dim),
                torch.nn.Sigmoid(),
            ]
        )
        self.decoder_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_features=code_dim, out_features=2000),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2000, out_features=500),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=500, out_features=500),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=500, out_features=input_shape),
                torch.nn.Sigmoid(),
            ]
        )
        for index, layer in enumerate(self.encoder_layers):
            if (index == (len(self.encoder_layers) - 1)) and isinstance(
                layer, torch.nn.Linear
            ):
                torch.nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            else:
                pass
        for index, layer in enumerate(self.decoder_layers):
            if (index == (len(self.decoder_layers) - 1)) and isinstance(
                layer, torch.nn.Linear
            ):
                torch.nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            else:
                pass
        self.train_loss = []
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.BCELoss().to(self.model_device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass by the model.

        Parameter
        ---------
        features: torch.Tensor
            The input features.

        Returns
        -------
        reconstruction: torch.Tensor
            The model output.
        """
        activations = {}
        for index, encoder_layer in enumerate(self.encoder_layers):
            if index == 0:
                activations[index] = encoder_layer(features)
            else:
                activations[index] = encoder_layer(activations[index - 1])
        code = activations[len(activations) - 1]
        activations = {}
        for index, decoder_layer in enumerate(self.decoder_layers):
            if index == 0:
                activations[index] = decoder_layer(code)
            else:
                activations[index] = decoder_layer(activations[index - 1])
        reconstruction = activations[len(activations) - 1]
        return reconstruction

    def fit(self, data_loader: object, epochs: int) -> None:
        """
        Trains the autoencoder model.

        Parameters
        ----------
        data_loader: torch.utils.dataloader.DataLoader
            The data loader object that consists of the data pipeline.
        epochs: int
            The number of epochs to train the model.
        """
        self.to(self.model_device)
        for epoch in range(epochs):
            epoch_loss = self.epoch_train(self, data_loader)
            if "cuda" in self.model_device.type:
                torch.cuda.empty_cache()
            self.train_loss.append(epoch_loss)
            print(f"epoch {epoch + 1}/{epochs} : mean loss = {self.train_loss[-1]:.6f}")

    def epoch_train(self, model: torch.nn.Module, data_loader: object) -> float:
        """
        Trains a model for one epoch.

        Parameters
        ----------
        model: torch.nn.Module
            The model to train.
        data_loader: torch.utils.dataloader.DataLoader
            The data loader object that consists of the data pipeline.

        Returns
        -------
        epoch_loss: float
            The epoch loss.
        """
        epoch_loss = 0
        for batch_features, _ in data_loader:
            batch_features = batch_features.view(batch_features.shape[0], -1)
            batch_features = batch_features.to(model.model_device)
            model.optimizer.zero_grad()
            outputs = model(batch_features)
            train_loss = model.criterion(outputs, batch_features)
            train_loss.backward()
            model.optimizer.step()
            epoch_loss += train_loss.item()
        epoch_loss /= len(data_loader)
        return epoch_loss

    def save_model(self, filename: str = "models/autoencoder.pth") -> None:
        """
        Exports the (presumably) trained autoencoder model.

        Parameter
        ---------
        filename: str
            The path and filename for the exported model.
        """
        print("[INFO] Exporting trained autoencoder model...")
        if not os.path.exists(os.path.dirname(filename)):
            os.mkdir(os.path.dirname(filename))
        torch.save(self.state_dict(), filename)
        print("[SUCCESS] Trained autoencoder model exported.")

    def load_model(self, filename: str = "models/autoencoder.pth") -> None:
        """
        Loads the trained autoencoder model.

        Parameter
        ---------
        filename: str
            The path to the trained autoencoder model.
        """
        print("[INFO] Loading the trained autoencoder model...")
        if os.path.isfile(filename):
            self.load_state_dict(torch.load(filename))
            print("[SUCCESS] Trained autoencoder ready for use.")
        else:
            print("[ERROR] Trained model not found.")

    def compute_latent_code(self, features: torch.Tensor) -> np.ndarray:
        """
        Computes the latent code representation for the input features.

        Parameter
        ---------
        features: torch.Tensor
            The input features whose latent code representation
            shall be computed.

        Returns
        -------
        latent_code: np.ndarray
            The latent code representation for the input features.
        """
        activations = {}
        for index, layer in enumerate(self.encoder_layers):
            if index == 0:
                activations[index] = layer(features)
            else:
                activations[index] = layer(activations.get(index - 1))
        latent_code = activations.get(len(activations) - 1)
        latent_code = latent_code.detach().numpy()
        return latent_code


class Clustering(object):
    def __init__(
        self,
        num_clusters: int,
        n_init: int = 10,
        epochs: int = 300,
        cores: int = 1,
        seed: int = None,
        tol: float = 1e-4,
        initialization: str = "random",
    ):
        """
        k-Means Clustering

        Parameters
        ----------
        num_clusters: int
            The number of clusters to form.
        n_init: int, optional, default: 10
            The number of times k-Means will be run
            with varying centroid seeds.
        epochs: int, optional, default: 300
            The maximum number of iterations
            k-Means will be run.
        cores: int, optional, default: 1
            The number of jobs to use for computing.
        seed: int, optional, default: None
            The random number generator seed.
            Set for reproducibility.
        tol: float, optional, default: 1e-4
            The tolerance with regards to inertia.
        initialization: str, optional, default: random
            The method for initialization.
        """
        self.model = KMeans(
            init=initialization,
            n_clusters=num_clusters,
            n_init=n_init,
            max_iter=epochs,
            n_jobs=cores,
            random_state=seed,
            tol=tol,
        )

    def train(self, features: np.ndarray) -> None:
        """
        Fit the dataset features.

        Parameter
        ---------
        features : np.ndarray
            The training instances to cluster.
        """
        self.model.fit(features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict the clusters to which the features belongs to.

        Parameter
        ---------
        features : np.ndarray
            The test instances to cluster.
        """
        cluster_predictions = self.model.predict(features)
        return cluster_predictions

    def benchmark(
        self, name: str, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[str, Dict]:
        """
        Returns the clustering performance results in str and dict format.

        The metrics used are as follows:
            1. Duration
            2. Adjusted RAND Score
            3. Normalized Mutual Information
            4. Davies-Bouldin Index
            5. Silhouette Score
            6. Calinski-Harabasz Score
            7. Clustering Accuracy

        Parameters
        ----------
        name: str
            The name of the benchmark.
        features: np.ndarray
            The test instances to cluster.
        labels: np.ndarray
            The test labels.

        Returns
        -------
        str
            The formatted string of the benchmark results.
        results: Dict
            The dictionary of benchmark results.
        """
        start_time = time.time()
        predictions = self.predict(features)

        results = {}

        results["name"] = name
        results["duration"] = time.time() - start_time
        results["ari"] = ari(labels_true=labels, labels_pred=predictions)
        results["nmi"] = nmi(labels_true=labels, labels_pred=predictions)
        results["dbi"] = davies_bouldin_score(features, predictions)
        results["silhouette"] = silhouette_score(
            features, predictions, metric="euclidean"
        )
        results["ch_score"] = calinski_harabasz_score(features, predictions)
        results["clustering_accuracy"] = clustering_accuracy(
            target=labels, prediction=predictions
        )

        return (
            "%-9s\t%.2fs\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f"
            % (
                results.get("name"),
                results.get("duration"),
                results.get("dbi"),
                results.get("silhouette"),
                results.get("ch_score"),
                results.get("nmi"),
                results.get("ari"),
                results.get("clustering_accuracy"),
            ),
            results,
        )

    def save_model(self, filename: str = "models/kmeans.pk") -> None:
        """
        Exports the fitted clustering model.

        Parameter
        ---------
        filename: str
            The path and filename for the exported model.
        """
        print("[INFO] Exporting clustering model...")
        if not os.path.exists(os.path.dirname(filename)):
            os.mkdir(os.path.dirname(filename))
        with open(filename, "wb") as model_file:
            pickle.dump(self.model, model_file)
        print("[SUCCESS] Clustering model exported.")
