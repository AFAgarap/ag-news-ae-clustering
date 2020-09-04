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
"""Module for clustering text using latent code"""
import sys

from ae_clustering.models import Autoencoder
from ae_clustering.utils import cluster_text, load_clustering_model, load_vectorizer


__author__ = "Abien Fred Agarap"


def main():
    text_input = sys.argv[1]
    vectorizer = load_vectorizer("data/vectorizer.pk")
    autoencoder = Autoencoder(input_shape=2000, code_dim=30)
    autoencoder.load_model("models/autoencoder.pth")
    kmeans = load_clustering_model("models/kmeans.pk")
    cluster_index = cluster_text(
        text=text_input,
        autoencoder_model=autoencoder,
        kmeans_model=kmeans,
        vectorizer=vectorizer,
    )
    print(f"Input text: {text_input}")
    print(f"Predicted cluster: {cluster_index.item()}")


if __name__ == "__main__":
    main()