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
"""Module for training autoencoder and clustering"""
from pt_datasets import load_dataset, create_dataloader

from ae_clustering.models import Autoencoder, Clustering
from ae_clustering.utils import (
    compute_latent_code,
    display_latent_code,
    display_results,
    export_vectorizer,
    set_global_seed,
)

__author__ = "Abien Fred Agarap"


def main():
    batch_size = 128
    clustering_epochs = 900
    code_dimensionality = 30
    epochs = 20
    learning_rate = 1e-3
    num_clusters = 4
    seed = 42
    set_global_seed(42)

    print("[INFO] Loading datasets...")
    train_data, test_data, vectorizer = load_dataset("ag_news", return_vectorizer=True)
    train_vectors = train_data.tensors[0]
    train_labels = train_data.tensors[1]
    test_vectors = test_data.tensors[0]
    test_labels = test_data.tensors[1]
    (train_vectors, train_labels, vectorizer) = load_dataset("data/ag_news.train")
    (test_vectors, test_labels, _) = load_dataset("data/ag_news.test")

    print("[INFO] Exporting vectorizer...")
    export_vectorizer(vectorizer=vectorizer, filename="data/vectorizer.pk")

    print("[INFO] Creating PyTorch data loader...")
    data_loader = create_dataloader(
        features=train_vectors, labels=train_labels, batch_size=batch_size
    )

    print("[INFO] Instantiating Autoencoder...")
    model = Autoencoder(
        input_shape=train_vectors.shape[1],
        code_dim=code_dimensionality,
        learning_rate=learning_rate,
    )

    print("[INFO] Training Autoencoder...")
    model.fit(data_loader=data_loader, epochs=epochs)
    model.save_model()

    print("[INFO] Switching from GPU to CPU for inference!")
    model = model.cpu()

    print("[INFO] Computing latent code...")
    train_latent_code = model.compute_latent_code(features=train_vectors)
    test_latent_code = model.compute_latent_code(features=test_vectors)

    print("[INFO] Clustering...")
    clustering = Clustering(
        num_clusters=num_clusters,
        seed=seed,
        cores=-1,
        initialization="k-means++",
        epochs=clustering_epochs,
    )
    clustering.train(features=train_latent_code)
    clustering.save_model()
    results, _ = clustering.benchmark(
        name="autoencoder", features=test_latent_code, labels=test_labels
    )
    display_results(results=results)
    display_latent_code(
        latent_code=test_latent_code[:1000],
        labels=test_labels[:1000],
        title="AG News Latent Code",
        seed=seed,
    )


if __name__ == "__main__":
    main()
