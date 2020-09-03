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

    def forward(self, features):
        """
        Defines the forward pass by the model.

        Parameter
        ---------
        features : torch.Tensor
            The input features.

        Returns
        -------
        reconstruction : torch.Tensor
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

    def fit(self, **kwargs):
        pass

    def epoch_train(self, model: torch.nn.Module, data_loader: object) -> float:
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
