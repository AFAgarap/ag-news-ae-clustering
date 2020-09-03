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
import torch

__author__ = "Abien Fred Agarap"


class Autoencoder(torch.nn.Module):
    def __init__(
        self,
        model_device: torch.device = torch.device("cpu"),
        input_shape: int = 784,
        code_dim: int = 128,
        learning_rate: float = 1e-4,
    ):
        """
        Constructs the autoencoder model.

        Parameters
        ----------
        model_device: torch.device
            The device to use for model computations.
        input_shape: int
            The dimensionality of the input features.
        code_dim: int
            The dimensionality of the latent code.
        learning_rate: float
            The learning rate to use for optimization.
        """
        super().__init__()
        self.model_device = model_device
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
        self.train_loss = []
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.BCELoss().to(self.model_device)

    def forward(self, **kwargs):
        pass

    def fit(self, **kwargs):
        pass

    def epoch_train(self, **kwargs):
        pass
